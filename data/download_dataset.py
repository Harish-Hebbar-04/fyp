"""
download_dataset.py
───────────────────
Downloads the SEP-28k stuttering dataset from Kaggle, organises audio clips,
and builds a unified CSV ready for the training pipeline.

SEP-28k Kaggle page:
  https://www.kaggle.com/datasets/abhranta/sep-28k-stuttering-detection

Usage
─────
  python data/download_dataset.py

Prerequisites
─────────────
1. Install the Kaggle CLI:  pip install kaggle
2. Place your Kaggle API token at  ~/.kaggle/kaggle.json
   (Download from https://www.kaggle.com/settings → API → Create New Token)
"""

import os
import sys
import zipfile
import shutil
import pandas as pd
from pathlib import Path

# ── allow imports from project root ──────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR, PROCESSED_DIR, LABEL2ID, SEP28K_COLUMNS

# ─────────────────────────────────────────────────────────────────────────
KAGGLE_DATASET = "abhranta/sep-28k-stuttering-detection"
ZIP_NAME       = "sep-28k-stuttering-detection.zip"
# ─────────────────────────────────────────────────────────────────────────


def download_dataset():
    """Download SEP-28k via the Kaggle API."""
    print("[1/4] Downloading SEP-28k dataset from Kaggle …")
    os.makedirs(RAW_DIR, exist_ok=True)

    try:
        import kaggle  # noqa: F401
    except ImportError:
        sys.exit("kaggle package not found – run:  pip install kaggle")

    os.system(
        f"kaggle datasets download -d {KAGGLE_DATASET} -p {RAW_DIR} --unzip"
    )
    print(f"      Dataset extracted to: {RAW_DIR}")


def locate_csv() -> Path:
    """Find the SEP-28k label CSV regardless of exact subfolder name."""
    for csv_path in RAW_DIR.rglob("*.csv"):
        df = pd.read_csv(csv_path, nrows=2)
        if any(col in df.columns for col in ["Prolongation", "Block", "SoundRep"]):
            print(f"      Found label CSV: {csv_path}")
            return csv_path
    sys.exit("Could not locate SEP-28k label CSV in the downloaded data.")


def build_label_csv(csv_path: Path) -> pd.DataFrame:
    """
    Convert SEP-28k's multi-label format into a single stutter-class label
    using the majority / priority rule:

        Block > Prolongation > SoundRep / WordRep > Interjection > Fluent
    """
    print("[2/4] Building unified label CSV …")
    df = pd.read_csv(csv_path)

    # Normalise column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Build audio file path column.
    # Supports both:
    # 1) SEP-28k Kaggle format (Show/EpId/ClipId)
    # 2) Pre-sliced format with a `filepath` column
    audio_root = csv_path.parent

    if "filepath" in df.columns:
        def clip_path_from_filepath(path_value):
            rel = str(path_value).strip().replace("\\", "/")
            candidates = [
                RAW_DIR / rel,
                audio_root / rel,
                audio_root.parent / rel,
            ]
            for candidate in candidates:
                if candidate.exists():
                    return str(candidate)
            return None

        df["audio_path"] = df["filepath"].apply(clip_path_from_filepath)
    else:
        def clip_path(row):
            """Resolve the mp3/wav file for each clip row."""
            show   = str(row.get("Show", "")).strip()
            ep_id  = str(row.get("EpId", "")).strip()
            clip_id = str(row.get("ClipId", "")).strip()
            # SEP-28k stores clips as  <Show>/<EpId>_<ClipId>.wav  (or .mp3)
            for ext in [".wav", ".mp3", ".flac"]:
                p = audio_root / show / f"{ep_id}_{clip_id}{ext}"
                if p.exists():
                    return str(p)
            return None

        df["audio_path"] = df.apply(clip_path, axis=1)

    # Drop rows with missing audio
    before = len(df)
    df = df[df["audio_path"].notna()].reset_index(drop=True)
    print(f"      Kept {len(df)} / {before} clips with resolved audio paths.")

    # ── Assign single label (priority order) ─────────────────────────────
    def assign_label(row):
        if row.get("Block", 0) == 1:
            return "Block"
        if row.get("Prolongation", 0) == 1:
            return "Prolongation"
        if row.get("SoundRep", 0) == 1 or row.get("WordRep", 0) == 1:
            return "Repetition"
        if row.get("Interjection", 0) == 1:
            return "Interjection"
        return "Fluent"

    df["label"]    = df.apply(assign_label, axis=1)
    df["label_id"] = df["label"].map(LABEL2ID)

    print("\n      Class distribution:")
    print(df["label"].value_counts().to_string())

    return df[["audio_path", "label", "label_id"]]


def save_processed(df: pd.DataFrame):
    """Save the cleaned CSV to the processed directory."""
    print("[3/4] Saving processed CSV …")
    out_path = PROCESSED_DIR / "sep28k_labels.csv"
    df.to_csv(out_path, index=False)
    print(f"      Saved → {out_path}")


def verify(df: pd.DataFrame):
    """Quick sanity check."""
    print("[4/4] Verification …")
    sample = df.sample(min(3, len(df)))
    for _, row in sample.iterrows():
        exists = Path(row["audio_path"]).exists()
        print(f"      {row['audio_path']}  |  label={row['label']}  |  file_exists={exists}")
    print("\n  Dataset ready.\n")


# ─────────────────────────────────────────────────────────────────────────

def main():
    # Check if dataset is already present
    try:
        csv_path = locate_csv()
        print("Dataset CSV already found, skipping download.")
    except SystemExit:
        # CSV not found, download
        download_dataset()
        csv_path = locate_csv()
    
    df       = build_label_csv(csv_path)
    save_processed(df)
    verify(df)


if __name__ == "__main__":
    main()
