"""
demo/scripts/generate_demo_certs.py

Generates 3 demo certificates for the hackathon presentation:
  1. genuine_demo.png     — clean certificate (should score 85-95)
  2. name_tampered.png    — name field edited (should score 20-35, FAKE)
  3. grade_tampered.png   — grade inflated (should score 25-40, FAKE)

Run: python demo/scripts/generate_demo_certs.py
Output: demo/certificates/
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pathlib import Path
import random
import math
import io

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops, ImageEnhance

OUT = Path("demo/certificates")
OUT.mkdir(parents=True, exist_ok=True)

W, H = 2480, 1754   # A4 landscape 300 DPI

# ── Font loader ────────────────────────────────────────────────────────────
def font(size, bold=False):
    candidates_bold = [
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf",
    ]
    candidates_reg = [
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    ]
    for c in (candidates_bold if bold else candidates_reg):
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    return ImageFont.load_default()


# ── Certificate renderer ───────────────────────────────────────────────────
def render_certificate(
    student_name:   str = "Rahul Kumar Sharma",
    institution:    str = "Delhi Technological University",
    degree:         str = "Bachelor of Technology",
    discipline:     str = "Computer Science & Engineering",
    roll_number:    str = "2019/DTU/CS/1234",
    issue_date:     str = "15 May 2023",
    grade:          str = "A+",
    cgpa:           str = "9.45",
    registrar:      str = "Prof. Ramesh Kumar",
    controller:     str = "Dr. Sunita Verma",
) -> Image.Image:

    BG      = (253, 252, 248)
    NAVY    = (10,  45,  90)
    GOLD    = (160, 120, 40)
    DARK    = (30,  30,  30)
    MID     = (80,  80,  80)

    img  = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # ── Outer border (double rect) ─────────────────────────────────────
    for inset, lw in [(18, 5), (34, 2)]:
        draw.rectangle([inset, inset, W-inset, H-inset], outline=NAVY, width=lw)

    # ── Corner ornaments ───────────────────────────────────────────────
    for cx, cy in [(90,90), (W-90,90), (90,H-90), (W-90,H-90)]:
        draw.ellipse([cx-30, cy-30, cx+30, cy+30], outline=GOLD, width=3)
        draw.ellipse([cx-18, cy-18, cx+18, cy+18], outline=GOLD, width=1)
        draw.line([(cx-42,cy),(cx-30,cy)], fill=GOLD, width=2)
        draw.line([(cx+30,cy),(cx+42,cy)], fill=GOLD, width=2)
        draw.line([(cx,cy-42),(cx,cy-30)], fill=GOLD, width=2)
        draw.line([(cx,cy+30),(cx,cy+42)], fill=GOLD, width=2)

    # ── Institution seal (top-left) ────────────────────────────────────
    seals = [(220, 180, 100, NAVY), (W-220, 180, 100, NAVY)]
    for sx2, sy2, r, col in seals:
        draw.ellipse([sx2-r, sy2-r, sx2+r, sy2+r], outline=col, width=5)
        draw.ellipse([sx2-r+12, sy2-r+12, sx2+r-12, sy2+r-12], outline=col, width=2)
        initials = "".join(w[0] for w in institution.split()[:3]).upper()
        draw.text((sx2, sy2), initials, font=font(48, True), fill=col, anchor="mm")
        draw.text((sx2, sy2+65), "OFFICIAL SEAL", font=font(20), fill=col, anchor="mm")

    cx = W // 2

    # ── Institution name ───────────────────────────────────────────────
    draw.text((cx, 195), institution.upper(),
              font=font(70, True), fill=NAVY, anchor="mm")
    draw.text((cx, 275), "Established · Accredited · Recognised",
              font=font(34), fill=GOLD, anchor="mm")

    # ── Divider ────────────────────────────────────────────────────────
    draw.line([(160, 315), (W-160, 315)], fill=NAVY, width=3)
    draw.line([(160, 322), (W-160, 322)], fill=GOLD, width=1)

    # ── CERTIFICATE OF DEGREE heading ─────────────────────────────────
    draw.text((cx, 390), "CERTIFICATE  OF  DEGREE",
              font=font(62, True), fill=DARK, anchor="mm")

    # ── Body text ──────────────────────────────────────────────────────
    y = 480
    draw.text((cx, y),      "This is to certify that",          font=font(40),     fill=MID,  anchor="mm")
    draw.text((cx, y+100),  student_name,                       font=font(76, True),fill=(10,10,100), anchor="mm")
    draw.text((cx, y+195),  f"Roll No.  {roll_number}",         font=font(38),     fill=MID,  anchor="mm")
    draw.text((cx, y+275),  "has successfully fulfilled all the requirements for the degree of",
              font=font(37), fill=MID, anchor="mm")
    draw.text((cx, y+365),  degree,                             font=font(56, True),fill=NAVY, anchor="mm")
    draw.text((cx, y+445),  f"in  {discipline}",               font=font(42),     fill=DARK, anchor="mm")

    # ── Grade ──────────────────────────────────────────────────────────
    draw.rectangle([cx-300, y+510, cx+300, y+575], outline=GOLD, width=2)
    draw.text((cx, y+542),
              f"Grade:  {grade}    ·    CGPA:  {cgpa} / 10.00",
              font=font(44, True), fill=(20, 120, 20), anchor="mm")

    # ── Date ───────────────────────────────────────────────────────────
    draw.text((cx, y+635), f"Date of Issue :  {issue_date}",
              font=font(38), fill=MID, anchor="mm")

    # ── Divider ────────────────────────────────────────────────────────
    draw.line([(160, H-200), (W-160, H-200)], fill=GOLD, width=1)

    # ── Signatures ────────────────────────────────────────────────────
    for name, title, x_pos in [
        (registrar,  "Registrar",                    350),
        ("Dr. A.K. Singh", "Principal",              cx),
        (controller, "Controller of Examinations",  W-350),
    ]:
        # Wavy signature line
        pts = []
        for i in range(0, 200, 10):
            pts.append((x_pos - 100 + i, H - 168 + random.randint(-7, 7)))
        for i in range(len(pts)-1):
            draw.line([pts[i], pts[i+1]], fill=(20,20,20), width=2)
        draw.line([(x_pos-100, H-152), (x_pos+100, H-152)], fill=NAVY, width=2)
        draw.text((x_pos, H-128), name,  font=font(26, True), fill=DARK, anchor="mm")
        draw.text((x_pos, H-96),  title, font=font(24),       fill=MID,  anchor="mm")

    return img


# ── Tampering functions ────────────────────────────────────────────────────

def tamper_name(img: Image.Image, new_name: str) -> Image.Image:
    """Simulate name swap: paint over original, write new name."""
    out  = img.copy()
    draw = ImageDraw.Draw(out)
    cx   = W // 2
    y    = 480 + 100

    # Paint over with slightly-off background (realistic tampering artifact)
    draw.rectangle([cx-700, y-55, cx+700, y+55], fill=(255, 254, 250))

    # Blur the smear region slightly
    arr    = np.array(out)
    region = arr[y-58:y+58, cx-703:cx+703]
    region = cv2.GaussianBlur(region, (9, 9), 0)
    arr[y-58:y+58, cx-703:cx+703] = region
    out = Image.fromarray(arr)
    draw = ImageDraw.Draw(out)

    # Write new name (slightly different shade — attacker didn't match exactly)
    draw.text((cx, y), new_name,
              font=font(76, True), fill=(12, 12, 108), anchor="mm")

    # Save as JPEG and reload (creates compression artifacts — key for ELA)
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=87)
    buf.seek(0)
    return Image.open(buf).copy()


def tamper_grade(img: Image.Image) -> Image.Image:
    """Simulate grade inflation: A → A+ and CGPA bump."""
    out  = img.copy()
    draw = ImageDraw.Draw(out)
    cx   = W // 2
    y    = 480 + 542

    draw.rectangle([cx-310, y-38, cx+310, y+38], fill=(253, 252, 248))
    draw.text((cx, y),
              "Grade:  A+    ·    CGPA:  9.95 / 10.00",
              font=font(44, True), fill=(20, 120, 20), anchor="mm")

    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=85)
    buf.seek(0)
    return Image.open(buf).copy()


def apply_realistic_scan_noise(img: Image.Image) -> Image.Image:
    """Add slight scan noise (tilt, gaussian noise) for realism."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 2.5, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    # Tiny tilt
    tilt = random.uniform(-0.8, 0.8)
    img = img.rotate(tilt, resample=Image.BICUBIC, fillcolor=(253, 252, 248))
    return img


# ── Generate all 3 demo certs ─────────────────────────────────────────────

def main():
    print("Generating demo certificates...")

    # 1. Genuine
    print("  [1/3] Genuine certificate...")
    genuine = render_certificate(
        student_name="Rahul Kumar Sharma",
        institution="Delhi Technological University",
        degree="Bachelor of Technology",
        discipline="Computer Science & Engineering",
        roll_number="2019/DTU/CS/1234",
        issue_date="15 May 2023",
        grade="A",
        cgpa="8.92",
        registrar="Prof. Ramesh Kumar",
        controller="Dr. Sunita Verma",
    )
    genuine = apply_realistic_scan_noise(genuine)
    genuine.save(OUT / "genuine_demo.png", "PNG")
    print(f"     → {OUT / 'genuine_demo.png'}")

    # 2. Name tampered
    print("  [2/3] Name-tampered certificate (FAKE)...")
    base = render_certificate(
        student_name="Priya Mehta",      # original name
        institution="Delhi Technological University",
        degree="Bachelor of Technology",
        discipline="Electronics & Communication",
        roll_number="2019/DTU/EC/5678",
        issue_date="20 June 2023",
        grade="B+",
        cgpa="7.81",
    )
    name_tampered = tamper_name(base, "Anjali Singh")  # swapped to different name
    name_tampered = apply_realistic_scan_noise(name_tampered)
    name_tampered.save(OUT / "name_tampered.png", "PNG")
    print(f"     → {OUT / 'name_tampered.png'}")

    # 3. Grade tampered
    print("  [3/3] Grade-tampered certificate (FAKE)...")
    base2 = render_certificate(
        student_name="Vikram Patel",
        institution="Indian Institute of Technology Delhi",
        degree="Master of Technology",
        discipline="Artificial Intelligence",
        roll_number="2021/IITD/AI/0099",
        issue_date="30 July 2023",
        grade="B",
        cgpa="6.50",
    )
    grade_tampered = tamper_grade(base2)
    grade_tampered = apply_realistic_scan_noise(grade_tampered)
    grade_tampered.save(OUT / "grade_tampered.png", "PNG")
    print(f"     → {OUT / 'grade_tampered.png'}")

    print(f"\n✓ All 3 demo certificates saved to {OUT}/")
    print("\nExpected verdicts during demo:")
    print("  genuine_demo.png   → GENUINE   (score ~85-95)")
    print("  name_tampered.png  → FAKE      (score ~15-35)")
    print("  grade_tampered.png → FAKE      (score ~20-40)")


if __name__ == "__main__":
    main()
