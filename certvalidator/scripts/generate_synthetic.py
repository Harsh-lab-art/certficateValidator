"""
scripts/generate_synthetic.py

Generates synthetic academic certificates for model training.
Produces two classes:
  - GENUINE  → clean, properly formatted certificates
  - FAKE     → tampered versions of genuine certs (name swap, date change,
               grade change, seal clone, re-compression artefacts)

Usage:
    python scripts/generate_synthetic.py \
        --count 1000 \
        --output ml/data/synthetic \
        --fake-ratio 0.5

Output structure:
    ml/data/synthetic/
        genuine/  *.png
        fake/     *.png
        labels.csv
"""

from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from faker import Faker
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from rich.console import Console
from rich.progress import track
import typer

console = Console()
fake = Faker(["en_IN", "en_US", "en_GB"])
app = typer.Typer()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CERT_W, CERT_H = 2480, 1754   # A4 landscape at 300 DPI
MARGIN = 120

DEGREES = [
    "Bachelor of Technology", "Master of Technology",
    "Bachelor of Science", "Master of Science",
    "Bachelor of Commerce", "Master of Business Administration",
    "Bachelor of Engineering", "Doctor of Philosophy",
    "Bachelor of Arts", "Bachelor of Computer Applications",
]

DISCIPLINES = [
    "Computer Science & Engineering", "Electronics & Communication",
    "Mechanical Engineering", "Civil Engineering",
    "Information Technology", "Data Science",
    "Artificial Intelligence", "Biotechnology",
]

INSTITUTIONS = [
    "Delhi Technological University",
    "Indian Institute of Technology",
    "Jawaharlal Nehru University",
    "University of Mumbai",
    "Anna University",
    "Bangalore University",
    "Calcutta University",
    "Osmania University",
    "Panjab University",
    "Gujarat University",
]

GRADES = ["A+", "A", "A-", "B+", "B", "First Division", "Distinction", "Pass with Merit"]
CGPA_RANGE = (5.5, 10.0)

BG_COLORS = [
    (255, 255, 255),    # pure white
    (252, 250, 245),    # warm off-white
    (248, 248, 255),    # cool white
    (255, 253, 245),    # cream
]

BORDER_COLORS = [
    (10, 60, 120),      # navy
    (80, 10, 10),       # maroon
    (10, 80, 10),       # dark green
    (100, 60, 0),       # dark gold
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CertData:
    student_name: str
    institution: str
    degree: str
    discipline: str
    roll_number: str
    issue_date: str
    grade: str
    cgpa: float
    registrar_name: str
    is_fake: bool = False
    tamper_type: Optional[str] = None   # "name" | "date" | "grade" | "seal"


# ---------------------------------------------------------------------------
# Font loader (fallback to PIL default if system fonts unavailable)
# ---------------------------------------------------------------------------

def _font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/System/Library/Fonts/Times.ttc",
        "C:/Windows/Fonts/times.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _font_bold(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/System/Library/Fonts/Times.ttc",
        "C:/Windows/Fonts/timesbd.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Certificate generator
# ---------------------------------------------------------------------------

class CertificateGenerator:

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def generate_genuine(self) -> tuple[Image.Image, CertData]:
        data = self._random_data()
        img = self._render(data)
        img = self._apply_realistic_noise(img)
        return img, data

    def generate_fake(self) -> tuple[Image.Image, CertData]:
        data = self._random_data()
        img = self._render(data)

        # Pick a tampering strategy
        strategy = random.choice(["name", "date", "grade", "seal", "multi"])
        img, data = self._tamper(img, data, strategy)
        img = self._apply_realistic_noise(img)
        data.is_fake = True
        data.tamper_type = strategy
        return img, data

    # ------------------------------------------------------------------ #
    #  Data generation                                                     #
    # ------------------------------------------------------------------ #

    def _random_data(self) -> CertData:
        issue_date = fake.date_between(start_date="-10y", end_date="today")
        return CertData(
            student_name=fake.name(),
            institution=random.choice(INSTITUTIONS),
            degree=random.choice(DEGREES),
            discipline=random.choice(DISCIPLINES),
            roll_number=f"{random.randint(10, 99)}{fake.bothify('???###').upper()}",
            issue_date=issue_date.strftime("%d %B %Y"),
            grade=random.choice(GRADES),
            cgpa=round(random.uniform(*CGPA_RANGE), 2),
            registrar_name=fake.name(),
        )

    # ------------------------------------------------------------------ #
    #  Rendering                                                           #
    # ------------------------------------------------------------------ #

    def _render(self, data: CertData) -> Image.Image:
        bg_color = random.choice(BG_COLORS)
        border_color = random.choice(BORDER_COLORS)

        img = Image.new("RGB", (CERT_W, CERT_H), bg_color)
        draw = ImageDraw.Draw(img)

        # Outer decorative border (double rect)
        bw = 18
        draw.rectangle([bw, bw, CERT_W - bw, CERT_H - bw],
                       outline=border_color, width=4)
        draw.rectangle([bw + 12, bw + 12, CERT_W - bw - 12, CERT_H - bw - 12],
                       outline=border_color, width=2)

        # Corner ornaments
        self._draw_corner_ornaments(draw, border_color)

        # Institution seal (circular placeholder)
        self._draw_seal(draw, img, data.institution, border_color)

        cx = CERT_W // 2

        # Header — institution name
        draw.text((cx, 200), data.institution,
                  font=_font_bold(72), fill=border_color, anchor="mm")

        # Subtitle
        draw.text((cx, 290), "Established · Recognised · Accredited",
                  font=_font(36), fill=(120, 100, 80), anchor="mm")

        # Divider line
        draw.line([(MARGIN * 2, 340), (CERT_W - MARGIN * 2, 340)],
                  fill=border_color, width=3)

        # "Certificate of" label
        draw.text((cx, 410), "CERTIFICATE OF DEGREE",
                  font=_font_bold(56), fill=(30, 30, 30), anchor="mm")

        # Body text
        body_y = 520
        draw.text((cx, body_y),
                  "This is to certify that",
                  font=_font(40), fill=(60, 60, 60), anchor="mm")

        # Student name (large, prominent — the most tampered field)
        draw.text((cx, body_y + 90), data.student_name,
                  font=_font_bold(68), fill=(10, 10, 80), anchor="mm")

        draw.text((cx, body_y + 185),
                  f"bearing Roll Number  {data.roll_number}",
                  font=_font(38), fill=(60, 60, 60), anchor="mm")

        draw.text((cx, body_y + 270),
                  "has successfully completed the requirements for the degree of",
                  font=_font(38), fill=(60, 60, 60), anchor="mm")

        draw.text((cx, body_y + 360), data.degree,
                  font=_font_bold(52), fill=border_color, anchor="mm")

        draw.text((cx, body_y + 440), f"in  {data.discipline}",
                  font=_font(42), fill=(40, 40, 40), anchor="mm")

        # Grade / CGPA
        draw.text((cx, body_y + 540),
                  f"with  {data.grade}  |  CGPA: {data.cgpa:.2f} / 10.00",
                  font=_font_bold(44), fill=(20, 100, 20), anchor="mm")

        # Date
        draw.text((cx, body_y + 630),
                  f"Date of Issue:  {data.issue_date}",
                  font=_font(38), fill=(60, 60, 60), anchor="mm")

        # Signatures
        self._draw_signatures(draw, data, border_color)

        return img

    def _draw_corner_ornaments(self, draw: ImageDraw.Draw, color: tuple):
        size = 60
        corners = [
            (MARGIN - 10, MARGIN - 10),
            (CERT_W - MARGIN - size + 10, MARGIN - 10),
            (MARGIN - 10, CERT_H - MARGIN - size + 10),
            (CERT_W - MARGIN - size + 10, CERT_H - MARGIN - size + 10),
        ]
        for cx, cy in corners:
            draw.ellipse([cx, cy, cx + size, cy + size], outline=color, width=3)
            draw.ellipse([cx + 8, cy + 8, cx + size - 8, cy + size - 8],
                         outline=color, width=1)

    def _draw_seal(self, draw: ImageDraw.Draw, img: Image.Image,
                   institution: str, color: tuple):
        # Position seal top-right
        sx, sy, sr = CERT_W - 320, 120, 120
        draw.ellipse([sx, sy, sx + sr * 2, sy + sr * 2], outline=color, width=4)
        draw.ellipse([sx + 10, sy + 10, sx + sr * 2 - 10, sy + sr * 2 - 10],
                     outline=color, width=2)
        # Institution initials in centre
        initials = "".join(w[0] for w in institution.split()[:3]).upper()
        draw.text((sx + sr, sy + sr), initials,
                  font=_font_bold(56), fill=color, anchor="mm")
        # "OFFICIAL SEAL" text
        draw.text((sx + sr, sy + sr * 2 + 20), "OFFICIAL SEAL",
                  font=_font(22), fill=color, anchor="mm")

    def _draw_signatures(self, draw: ImageDraw.Draw, data: CertData, color: tuple):
        y_sig = CERT_H - 200
        # Left: Registrar
        self._draw_signature_block(draw, MARGIN + 200, y_sig,
                                   data.registrar_name, "Registrar", color)
        # Right: Controller of Examinations
        self._draw_signature_block(draw, CERT_W - MARGIN - 200, y_sig,
                                   fake.name(), "Controller of Examinations", color)
        # Centre: Principal
        self._draw_signature_block(draw, CERT_W // 2, y_sig,
                                   fake.name(), "Principal", color)

    def _draw_signature_block(self, draw: ImageDraw.Draw, x: int, y: int,
                              name: str, title: str, color: tuple):
        # Wavy signature line (simulated)
        pts = []
        for i in range(0, 160, 8):
            pts.append((x - 80 + i, y - 10 + random.randint(-6, 6)))
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=(20, 20, 20), width=2)

        draw.line([(x - 80, y), (x + 80, y)], fill=color, width=2)
        draw.text((x, y + 20), name, font=_font_bold(28), fill=(30, 30, 30), anchor="mm")
        draw.text((x, y + 55), title, font=_font(26), fill=(80, 80, 80), anchor="mm")

    # ------------------------------------------------------------------ #
    #  Realistic noise (makes it look scanned/photographed)               #
    # ------------------------------------------------------------------ #

    def _apply_realistic_noise(self, img: Image.Image) -> Image.Image:
        noise_type = random.choice(["gaussian", "slight_blur", "jpeg_artefact", "none"])

        if noise_type == "gaussian":
            arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, random.uniform(1, 4), arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        elif noise_type == "slight_blur":
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))

        elif noise_type == "jpeg_artefact":
            buf = __import__("io").BytesIO()
            img.save(buf, "JPEG", quality=random.randint(75, 92))
            buf.seek(0)
            img = Image.open(buf).copy()

        # Random very slight rotation (scanning tilt)
        tilt = random.uniform(-1.5, 1.5)
        img = img.rotate(tilt, resample=Image.BICUBIC, expand=False,
                         fillcolor=random.choice(BG_COLORS))
        return img

    # ------------------------------------------------------------------ #
    #  Tampering strategies                                                #
    # ------------------------------------------------------------------ #

    def _tamper(self, img: Image.Image, data: CertData,
                strategy: str) -> tuple[Image.Image, CertData]:
        if strategy == "name":
            return self._tamper_name(img, data)
        elif strategy == "date":
            return self._tamper_date(img, data)
        elif strategy == "grade":
            return self._tamper_grade(img, data)
        elif strategy == "seal":
            return self._tamper_seal(img, data)
        elif strategy == "multi":
            # Two tamperings on same certificate
            strategies = random.sample(["name", "date", "grade"], 2)
            for s in strategies:
                img, data = self._tamper(img, data, s)
            return img, data
        return img, data

    def _tamper_name(self, img: Image.Image, data: CertData) -> tuple[Image.Image, CertData]:
        """
        Paint-over the student name region with background colour, then
        write a different name. This leaves JPEG recompression artefacts
        visible in ELA — the key forensic signal.
        """
        draw = ImageDraw.Draw(img)
        cx = CERT_W // 2
        y_name = 520 + 90   # matches rendering y-position

        # Estimate text bounding box for old name
        font = _font_bold(68)
        bbox_w = len(data.student_name) * 36   # rough estimate
        rect = [cx - bbox_w, y_name - 45, cx + bbox_w, y_name + 45]

        # Smear over with slightly-off-white (tamperer didn't colour-match perfectly)
        smear_color = (random.randint(245, 255),) * 3
        draw.rectangle(rect, fill=smear_color)

        # Add subtle blur to smear region — realistic
        arr = np.array(img)
        region = arr[y_name - 50:y_name + 50, cx - bbox_w:cx + bbox_w]
        region = cv2.GaussianBlur(region, (7, 7), 0)
        arr[y_name - 50:y_name + 50, cx - bbox_w:cx + bbox_w] = region
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)

        # Write new (fake) name — slight colour/font inconsistency
        new_name = fake.name()
        data.student_name = new_name + " [TAMPERED]"    # for label
        tamper_fill = (random.randint(0, 20), random.randint(0, 20),
                       random.randint(60, 100))
        draw.text((cx, y_name), new_name,
                  font=font, fill=tamper_fill, anchor="mm")
        return img, data

    def _tamper_date(self, img: Image.Image, data: CertData) -> tuple[Image.Image, CertData]:
        draw = ImageDraw.Draw(img)
        cx = CERT_W // 2
        y_date = 520 + 630

        rect = [cx - 400, y_date - 30, cx + 400, y_date + 35]
        draw.rectangle(rect, fill=(255, 255, 255))

        new_date = fake.date_between(start_date="-2y", end_date="today").strftime("%d %B %Y")
        data.issue_date = new_date
        draw.text((cx, y_date),
                  f"Date of Issue:  {new_date}",
                  font=_font(38), fill=(60, 60, 60), anchor="mm")
        return img, data

    def _tamper_grade(self, img: Image.Image, data: CertData) -> tuple[Image.Image, CertData]:
        draw = ImageDraw.Draw(img)
        cx = CERT_W // 2
        y_grade = 520 + 540

        rect = [cx - 500, y_grade - 35, cx + 500, y_grade + 38]
        draw.rectangle(rect, fill=(255, 255, 253))

        new_grade = "A+"
        new_cgpa = round(random.uniform(9.0, 10.0), 2)
        data.grade = new_grade
        data.cgpa = new_cgpa
        draw.text((cx, y_grade),
                  f"with  {new_grade}  |  CGPA: {new_cgpa:.2f} / 10.00",
                  font=_font_bold(44), fill=(20, 100, 20), anchor="mm")
        return img, data

    def _tamper_seal(self, img: Image.Image, data: CertData) -> tuple[Image.Image, CertData]:
        """Clone the seal from one position and paste at a slightly different offset."""
        arr = np.array(img)
        sx, sy, sr = CERT_W - 320, 120, 120
        seal_region = arr[sy:sy + sr * 2 + 30, sx:sx + sr * 2 + 30].copy()

        # Paste at slightly shifted position (clone stamp)
        offset_x = random.randint(-15, 15)
        offset_y = random.randint(-15, 15)
        ny = sy + offset_y
        nx = sx + offset_x
        ny = max(0, min(ny, arr.shape[0] - seal_region.shape[0]))
        nx = max(0, min(nx, arr.shape[1] - seal_region.shape[1]))
        arr[ny:ny + seal_region.shape[0], nx:nx + seal_region.shape[1]] = seal_region

        return Image.fromarray(arr), data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    count: int = typer.Option(500, help="Total certificates to generate"),
    output: Path = typer.Option(Path("ml/data/synthetic"), help="Output directory"),
    fake_ratio: float = typer.Option(0.5, help="Fraction of fake certificates (0.0–1.0)"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
):
    """Generate synthetic training certificates (genuine + fake)."""
    genuine_dir = output / "genuine"
    fake_dir = output / "fake"
    genuine_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    gen = CertificateGenerator(seed=seed)
    n_fake = int(count * fake_ratio)
    n_genuine = count - n_fake

    rows = []

    console.print(f"[cyan]Generating {n_genuine} genuine certificates...[/cyan]")
    for i in track(range(n_genuine), description="Genuine"):
        img, data = gen.generate_genuine()
        filename = f"genuine_{i:05d}.png"
        img.save(genuine_dir / filename, "PNG")
        rows.append({
            "filename": f"genuine/{filename}",
            "label": 0,
            "is_fake": False,
            "tamper_type": "",
            "student_name": data.student_name,
            "institution": data.institution,
            "degree": data.degree,
            "issue_date": data.issue_date,
            "grade": data.grade,
            "cgpa": data.cgpa,
        })

    console.print(f"[cyan]Generating {n_fake} fake certificates...[/cyan]")
    for i in track(range(n_fake), description="Fake"):
        img, data = gen.generate_fake()
        filename = f"fake_{i:05d}.png"
        img.save(fake_dir / filename, "PNG")
        rows.append({
            "filename": f"fake/{filename}",
            "label": 1,
            "is_fake": True,
            "tamper_type": data.tamper_type or "",
            "student_name": data.student_name,
            "institution": data.institution,
            "degree": data.degree,
            "issue_date": data.issue_date,
            "grade": data.grade,
            "cgpa": data.cgpa,
        })

    # Write labels CSV
    labels_path = output / "labels.csv"
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    console.print(
        f"\n[green]Done![/green] "
        f"{n_genuine} genuine + {n_fake} fake → {output}\n"
        f"Labels: {labels_path}"
    )


if __name__ == "__main__":
    app()
