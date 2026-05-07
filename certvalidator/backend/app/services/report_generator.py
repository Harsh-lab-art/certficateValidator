"""
backend/app/services/report_generator.py

Forensic PDF report generator using ReportLab.

Produces a professional multi-page PDF that includes:
  Page 1: Summary — verdict banner, trust score dial, sub-scores
  Page 2: Original certificate + GradCAM heatmap side by side
  Page 3: Field extraction table + NLP reasoning + audit trail

This is the downloadable report judges see when you click "Export PDF".
"""
from __future__ import annotations

import io
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage, PageBreak,
    )
    from reportlab.graphics.shapes import Drawing, Wedge, Circle, String
    from reportlab.graphics import renderPDF
    REPORTLAB_AVAILABLE = True

    # ── Colour palette ────────────────────────────────────────────────────
    C_NAVY      = colors.HexColor("#0f172a")
    C_INDIGO    = colors.HexColor("#4f46e5")
    C_GENUINE   = colors.HexColor("#16a34a")
    C_FAKE      = colors.HexColor("#dc2626")
    C_SUSPICIOUS= colors.HexColor("#d97706")
    C_LIGHTGRAY = colors.HexColor("#f1f5f9")
    C_MIDGRAY   = colors.HexColor("#94a3b8")
    C_WHITE     = colors.white
    C_BLACK     = colors.HexColor("#0f172a")

except ImportError:
    REPORTLAB_AVAILABLE = False
    C_NAVY = C_INDIGO = C_GENUINE = C_FAKE = C_SUSPICIOUS = None
    C_LIGHTGRAY = C_MIDGRAY = C_WHITE = C_BLACK = None


def verdict_color(verdict: str):
    return {
        "GENUINE":    C_GENUINE,
        "FAKE":       C_FAKE,
        "SUSPICIOUS": C_SUSPICIOUS,
    }.get(verdict, C_MIDGRAY)


# ── Trust score dial (drawn with ReportLab graphics) ─────────────────────

def _make_score_dial(score: float, verdict: str, size: float = 120) -> Drawing:
    """Draw a semicircular trust score gauge."""
    d = Drawing(size, size * 0.7)
    cx, cy = size / 2, size * 0.55
    r_outer = size * 0.42
    r_inner = size * 0.28

    # Background arc
    for i in range(180):
        angle = i
        frac  = i / 180
        if frac < 0.45:
            c = C_FAKE
        elif frac < 0.75:
            c = C_SUSPICIOUS
        else:
            c = C_GENUINE
        w = Wedge(cx, cy, r_outer, angle, angle + 1.5, innerRadiusFraction=r_inner / r_outer)
        w.fillColor  = c
        w.strokeColor= None
        d.add(w)

    # Score needle
    import math
    needle_angle = 180 - (score / 100 * 180)
    rad = math.radians(needle_angle)
    nx  = cx + r_inner * 0.9 * math.cos(rad)
    ny  = cy + r_inner * 0.9 * math.sin(rad)
    from reportlab.graphics.shapes import Line
    l = Line(cx, cy, nx, ny)
    l.strokeColor = C_BLACK
    l.strokeWidth = 2
    d.add(l)

    # Centre circle
    c_dot = Circle(cx, cy, size * 0.05)
    c_dot.fillColor   = C_BLACK
    c_dot.strokeColor = None
    d.add(c_dot)

    # Score text
    s = String(cx, cy - size * 0.22, f"{score:.0f}", fontSize=size * 0.16,
               fillColor=verdict_color(verdict), textAnchor="middle")
    d.add(s)
    s2 = String(cx, cy - size * 0.35, "/ 100", fontSize=size * 0.08,
                fillColor=C_MIDGRAY, textAnchor="middle")
    d.add(s2)

    return d


# ── Main generator ────────────────────────────────────────────────────────

def generate_pdf_report(result, output_path: str) -> str:
    """
    Generate a full forensic PDF report.

    result      : VerificationResult from inference.py
    output_path : where to write the PDF

    Returns output_path on success.
    """
    if not REPORTLAB_AVAILABLE:
        # Write a minimal placeholder
        with open(output_path, "wb") as f:
            f.write(b"%PDF-1.4 placeholder - install reportlab")
        return output_path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=15*mm, bottomMargin=15*mm,
        leftMargin=20*mm, rightMargin=20*mm,
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Page 1: Summary ───────────────────────────────────────────────
    # Header bar
    header_data = [[
        Paragraph(
            f'<font color="#ffffff"><b>CertValidator</b> — Forensic Report</font>',
            ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=14,
                           textColor=C_WHITE, alignment=TA_LEFT)
        ),
        Paragraph(
            f'<font color="#94a3b8">{datetime.now().strftime("%d %B %Y  %H:%M")}</font>',
            ParagraphStyle("ts", fontName="Helvetica", fontSize=9,
                           textColor=C_MIDGRAY, alignment=TA_RIGHT)
        ),
    ]]
    header_tbl = Table(header_data, colWidths=[120*mm, 50*mm])
    header_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), C_NAVY),
        ("TOPPADDING",  (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 8*mm))

    # Verdict banner
    v_color = verdict_color(result.verdict)
    verdict_text = {
        "GENUINE":    "CERTIFICATE VERIFIED — GENUINE",
        "FAKE":       "CERTIFICATE REJECTED — FAKE / TAMPERED",
        "SUSPICIOUS": "CERTIFICATE FLAGGED — SUSPICIOUS",
    }.get(result.verdict, result.verdict)

    banner = Table([[
        Paragraph(
            f'<font color="#ffffff"><b>{verdict_text}</b></font>',
            ParagraphStyle("v", fontName="Helvetica-Bold", fontSize=16,
                           textColor=C_WHITE, alignment=TA_CENTER)
        )
    ]], colWidths=[170*mm])
    banner.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), v_color),
        ("TOPPADDING",   (0,0), (-1,-1), 10),
        ("BOTTOMPADDING",(0,0), (-1,-1), 10),
        ("ROUNDEDCORNERS",(0,0),(-1,-1), 6),
    ]))
    story.append(banner)
    story.append(Spacer(1, 6*mm))

    # Trust score dial + sub-scores
    try:
        dial = _make_score_dial(result.trust_score, result.verdict)
        score_row = [[
            RLImage(io.BytesIO(dial.asString("png")), width=50*mm, height=35*mm)
            if hasattr(dial, "asString") else
            Paragraph(f"<b>{result.trust_score:.0f}/100</b>",
                      ParagraphStyle("sc", fontSize=28, alignment=TA_CENTER)),
            _sub_scores_table(result),
        ]]
    except Exception:
        score_row = [[
            Paragraph(
                f'<font size="28"><b>{result.trust_score:.0f}/100</b></font>',
                ParagraphStyle("sc", fontSize=28, alignment=TA_CENTER)
            ),
            _sub_scores_table(result),
        ]]

    score_tbl = Table(score_row, colWidths=[60*mm, 110*mm])
    score_tbl.setStyle(TableStyle([
        ("VALIGN",  (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1), 4),
    ]))
    story.append(score_tbl)
    story.append(Spacer(1, 5*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_LIGHTGRAY))
    story.append(Spacer(1, 4*mm))

    # Explanation paragraph
    story.append(Paragraph(
        result.explanation,
        ParagraphStyle("exp", fontName="Helvetica", fontSize=10,
                       textColor=C_BLACK, leading=14)
    ))
    story.append(Spacer(1, 6*mm))

    # ── Field breakdown table ─────────────────────────────────────────
    story.append(Paragraph(
        "<b>Field Extraction Results</b>",
        ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12, textColor=C_NAVY)
    ))
    story.append(Spacer(1, 3*mm))

    field_rows = [["Field", "Extracted Value", "Confidence", "Status"]]
    for fs in (result.field_scores or []):
        conf    = fs.get("confidence", 0)
        flagged = fs.get("flagged", False)
        status_text = "FLAGGED" if flagged else "OK"
        status_color= C_FAKE if flagged else C_GENUINE
        field_rows.append([
            fs.get("field", "").replace("_", " ").title(),
            str(fs.get("value") or "—")[:45],
            f"{conf:.0%}",
            Paragraph(f'<font color="{status_color.hexval() if hasattr(status_color,"hexval") else "#000000"}">'
                      f'<b>{status_text}</b></font>',
                      ParagraphStyle("st", fontSize=9)),
        ])

    field_tbl = Table(field_rows, colWidths=[35*mm, 80*mm, 25*mm, 25*mm])
    field_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), C_NAVY),
        ("TEXTCOLOR",     (0,0), (-1,0), C_WHITE),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [C_WHITE, C_LIGHTGRAY]),
        ("GRID",          (0,0), (-1,-1), 0.3, C_MIDGRAY),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 5),
    ]))
    story.append(field_tbl)
    story.append(Spacer(1, 6*mm))

    # ── NLP Reasoning ────────────────────────────────────────────────
    story.append(Paragraph(
        "<b>AI Forensic Reasoning</b>",
        ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12, textColor=C_NAVY)
    ))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        result.nlp_reasoning or "Reasoning not available.",
        ParagraphStyle("reasoning", fontName="Helvetica", fontSize=9,
                       textColor=C_BLACK, leading=13,
                       borderColor=C_LIGHTGRAY, borderWidth=1,
                       borderPadding=6, backColor=C_LIGHTGRAY)
    ))
    story.append(Spacer(1, 5*mm))

    # ── Tamper regions ────────────────────────────────────────────────
    if result.tamper_regions:
        story.append(Paragraph(
            f"<b>Tamper Regions Detected ({len(result.tamper_regions)})</b>",
            ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12, textColor=C_FAKE)
        ))
        region_rows = [["Region", "Position (x,y)", "Size (w×h)", "Confidence"]]
        for i, r in enumerate(result.tamper_regions, 1):
            region_rows.append([
                f"Region {i}",
                f"({r['x']}, {r['y']})",
                f"{r['width']} × {r['height']} px",
                f"{r['confidence']:.0%}",
            ])
        r_tbl = Table(region_rows, colWidths=[30*mm, 45*mm, 50*mm, 40*mm])
        r_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), C_FAKE),
            ("TEXTCOLOR",  (0,0), (-1,0), C_WHITE),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 9),
            ("GRID",       (0,0), (-1,-1), 0.3, C_MIDGRAY),
            ("TOPPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(Spacer(1, 3*mm))
        story.append(r_tbl)
        story.append(Spacer(1, 5*mm))

    # ── Page 2: Heatmap ───────────────────────────────────────────────
    if result.heatmap_path and Path(result.heatmap_path).exists():
        story.append(PageBreak())
        story.append(Paragraph(
            "<b>GradCAM Heatmap — Tamper Detection Overlay</b>",
            ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12, textColor=C_NAVY)
        ))
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph(
            "Red/yellow regions show where the forgery detector found the strongest tampering signal. "
            "Blue regions are low-suspicion areas.",
            ParagraphStyle("cap", fontName="Helvetica", fontSize=9, textColor=C_MIDGRAY)
        ))
        story.append(Spacer(1, 4*mm))
        try:
            story.append(RLImage(result.heatmap_path, width=170*mm, height=110*mm))
        except Exception:
            story.append(Paragraph("Heatmap image unavailable.",
                                   ParagraphStyle("err", fontSize=9, textColor=C_MIDGRAY)))

    # ── Audit trail ───────────────────────────────────────────────────
    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width="100%", thickness=0.3, color=C_LIGHTGRAY))
    story.append(Spacer(1, 3*mm))

    audit_style = ParagraphStyle("audit", fontName="Helvetica", fontSize=8,
                                  textColor=C_MIDGRAY)
    story.append(Paragraph(
        f"Verification ID: {result.verification_id}  |  "
        f"File hash (SHA-256): {result.file_hash[:32]}...  |  "
        f"Processing time: {result.processing_time_s:.2f}s  |  "
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        audit_style
    ))
    story.append(Paragraph(
        f"Models: {' | '.join(f'{k}: {v}' for k,v in (result.model_versions or {}).items())}",
        audit_style
    ))

    doc.build(story)
    return output_path


def _sub_scores_table(result):
    """Build the 3-row sub-score breakdown table."""
    from reportlab.platypus import Table, TableStyle
    rows = [
        ["Component",           "Score",   "Weight"],
        ["Forgery detector",    f"{(1-result.forgery_score):.0%}",    "45%"],
        ["Field confidence",    f"{result.field_confidence:.0%}",     "35%"],
        ["NLP reasoning",       f"{(1-result.nlp_anomaly_score):.0%}","20%"],
        ["Institution match",   "YES" if result.institution_matched else "NO", "bonus"],
    ]
    tbl = Table(rows, colWidths=[50*mm, 30*mm, 25*mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), C_NAVY),
        ("TEXTCOLOR",   (0,0), (-1,0), C_WHITE),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, C_LIGHTGRAY]),
        ("GRID",        (0,0), (-1,-1), 0.3, C_MIDGRAY),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
    ]))
    return tbl


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """Quick test — generate a dummy report."""
    from types import SimpleNamespace
    dummy = SimpleNamespace(
        verification_id="test-1234", verdict="GENUINE", trust_score=87.4,
        explanation="Certificate appears genuine. All fields consistent.",
        forgery_score=0.08, field_confidence=0.93, nlp_anomaly_score=0.11,
        institution_matched=True,
        field_scores=[
            {"field":"student_name","value":"Rahul Sharma","confidence":0.97,"flagged":False},
            {"field":"institution", "value":"IIT Delhi",   "confidence":0.95,"flagged":False},
            {"field":"grade",       "value":"9.2/10",      "confidence":0.90,"flagged":False},
        ],
        tamper_regions=[],
        nlp_reasoning="All extracted fields are internally consistent. No anomalies detected.",
        heatmap_path=None,
        file_hash="abc123" * 10,
        processing_time_s=4.23,
        model_versions={"forgery":"efficientnet-b4","layout":"layoutlmv3","nlp":"heuristic"},
    )
    out = generate_pdf_report(dummy, "/tmp/test_report.pdf")
    print(f"Report written to {out}")
