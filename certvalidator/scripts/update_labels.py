"""
scripts/update_labels.py

Scans ml/data/synthetic/genuine/ and ml/data/synthetic/fake/ for any image
or PDF files that are NOT yet in labels.csv and appends them automatically.

Run this before training whenever you add new data:
    python scripts/update_labels.py

Or let train.py call it automatically (it does by default).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

SYNTHETIC_DIR = Path("ml/data/synthetic")
LABELS_CSV    = SYNTHETIC_DIR / "labels.csv"
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".pdf"}

FIELDNAMES = [
    "filename", "label", "is_fake", "tamper_type",
    "student_name", "institution", "degree",
    "issue_date", "grade", "cgpa",
]


def update_labels(data_dir: Path = SYNTHETIC_DIR, verbose: bool = True) -> int:
    """
    Scan genuine/ and fake/ folders, add any missing files to labels.csv.
    Returns the number of new rows added.
    """
    labels_path = data_dir / "labels.csv"

    # Load existing entries
    existing_files: set[str] = set()
    existing_rows: list[dict] = []

    if labels_path.exists():
        with open(labels_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append(row)
                existing_files.add(Path(row["filename"]).name)

    new_rows: list[dict] = []

    for folder, label, is_fake, tamper in [
        ("genuine", "0", "False", ""),
        ("fake",    "1", "True",  "real_fake"),
    ]:
        folder_path = data_dir / folder
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
            continue

        for f in sorted(folder_path.iterdir()):
            if f.suffix.lower() not in SUPPORTED_EXT:
                continue
            if f.name in existing_files:
                continue  # already in CSV

            new_rows.append({
                "filename":     f"{folder}/{f.name}",
                "label":        label,
                "is_fake":      is_fake,
                "tamper_type":  tamper,
                "student_name": "",
                "institution":  "",
                "degree":       "",
                "issue_date":   "",
                "grade":        "",
                "cgpa":         "0.0",
            })
            existing_files.add(f.name)

    if not new_rows:
        if verbose:
            print(f"[update_labels] labels.csv is up to date — {len(existing_rows)} entries.")
        return 0

    # Write updated CSV (existing + new)
    all_rows = existing_rows + new_rows
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    if verbose:
        genuine_new = sum(1 for r in new_rows if r["label"] == "0")
        fake_new    = sum(1 for r in new_rows if r["label"] == "1")
        print(f"[update_labels] Added {len(new_rows)} new entries "
              f"({genuine_new} genuine, {fake_new} fake). "
              f"Total: {len(all_rows)}")

    return len(new_rows)


if __name__ == "__main__":
    added = update_labels()
    sys.exit(0)
