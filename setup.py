"""
setup.py
─────────
One-command full project setup:
  1. Validates Kaggle credentials are configured
  2. Downloads the SEP-28k dataset
  3. Preprocesses labels into a clean CSV
  4. Trains the model (audio-only, 10 epochs as a quick start)

Run:
  python setup.py

After this, run the app:
  python run.py
"""

import sys
import json
import subprocess
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


def step(msg: str):
    print(f"\n{'─'*60}")
    print(f"  {msg}")
    print('─'*60)


def run(cmd: list, **kwargs):
    """Run a subprocess command and stream output."""
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result


# ─────────────────────────────────────────────────────────────────────────
# Step 1: Check Kaggle credentials
# ─────────────────────────────────────────────────────────────────────────

def check_kaggle():
    step("Step 1 / 3  –  Checking Kaggle credentials")

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    if not kaggle_json.exists():
        print(f"""
  ✗  Kaggle credentials not found at:
       {kaggle_json}

  How to fix:
    1. Go to https://www.kaggle.com/settings
    2. Click API → Create New Token
    3. Save the downloaded kaggle.json to:
         {kaggle_json}
    4. Re-run:  python setup.py
""")
        sys.exit(1)

    try:
        creds = json.loads(kaggle_json.read_text(encoding="utf-8"))
        if creds.get("username") == "YOUR_KAGGLE_USERNAME":
            print(f"""
  ✗  kaggle.json still contains the template values.

  How to fix:
    1. Go to https://www.kaggle.com/settings
    2. Click API → Create New Token
    3. Replace the file at:
         {kaggle_json}
    4. Re-run:  python setup.py
""")
            sys.exit(1)
    except Exception:
        print("  ✗  Could not parse kaggle.json – check the file format.")
        sys.exit(1)

    print(f"  ✓  Kaggle credentials found  (user: {creds.get('username')})")


# ─────────────────────────────────────────────────────────────────────────
# Step 2: Download dataset
# ─────────────────────────────────────────────────────────────────────────

def download_dataset():
    step("Step 2 / 3  –  Downloading SEP-28k dataset from Kaggle")

    processed_csv = ROOT / "data" / "processed" / "sep28k_labels.csv"
    if processed_csv.exists():
        print(f"  ✓  Dataset already downloaded  ({processed_csv})")
        print("     Delete the file and re-run to re-download.")
        return

    run([PYTHON, str(ROOT / "data" / "download_dataset.py")])
    print("  ✓  Dataset ready.")


# ─────────────────────────────────────────────────────────────────────────
# Step 3: Train the model
# ─────────────────────────────────────────────────────────────────────────

def train_model():
    step("Step 3 / 3  –  Training the stutter detection model")

    checkpoint = ROOT / "models" / "saved" / "stutter_model.pt"
    if checkpoint.exists():
        print(f"  ✓  Model checkpoint already exists  ({checkpoint})")
        print("     Delete the file and re-run to re-train.")
        return

    print("  Running:  python training/train.py --mode audio_only --epochs 10")
    print("  (Use --epochs 20 in training/train.py for better accuracy)\n")

    run([
        PYTHON, str(ROOT / "training" / "train.py"),
        "--mode",       "audio_only",
        "--epochs",     "10",
        "--batch",      "16",
        "--checkpoint", "stutter_model",
    ])

    if checkpoint.exists():
        print(f"\n  ✓  Model saved → {checkpoint}")
    else:
        print("  ✗  Training completed but checkpoint not found.")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  Stutter Detection System  –  Full Setup")
    print("="*60)

    check_kaggle()
    download_dataset()
    train_model()

    print("\n" + "="*60)
    print("  Setup complete!  Run the app:")
    print("    python run.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
