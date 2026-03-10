# ============================================================
#  Stutter Detection System – One-Command Bootstrap
#  Usage (from project root):
#    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
#    .\bootstrap.ps1
#
#  Optionally train after setup:
#    .\bootstrap.ps1 -Train
# ============================================================

param(
    [switch]$Train   # pass -Train to also run model training after setup
)

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

function Step($msg) {
    Write-Host ""
    Write-Host ("─" * 60) -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host ("─" * 60) -ForegroundColor Cyan
}

# ── 1. Check Python ─────────────────────────────────────────
Step "1/5  Checking Python installation"
try {
    $pyver = python --version 2>&1
    Write-Host "  Found: $pyver" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Python not found. Install from https://python.org (3.10+)" -ForegroundColor Red
    exit 1
}

# ── 2. Create virtual environment ───────────────────────────
Step "2/5  Creating virtual environment (.venv)"
if (-Not (Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "  Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "  .venv already exists, skipping." -ForegroundColor Yellow
}

$PY  = ".\\.venv\\Scripts\\python.exe"
$PIP = ".\\.venv\\Scripts\\pip.exe"

# ── 3. Install dependencies ──────────────────────────────────
Step "3/5  Installing dependencies from requirements.txt"
& $PIP install --upgrade pip --quiet
& $PIP install -r requirements.txt
Write-Host "  Dependencies installed." -ForegroundColor Green

# ── 4. Prepare dataset ───────────────────────────────────────
Step "4/5  Preparing SEP-28k dataset"

$archiveZip  = "$ROOT\data\raw\archive.zip"
$processedCsv = "$ROOT\data\processed\sep28k_labels.csv"
$kaggleCreds = "$env:USERPROFILE\.kaggle\kaggle.json"

if (Test-Path $processedCsv) {
    Write-Host "  Dataset already processed – skipping." -ForegroundColor Green
} elseif (Test-Path $archiveZip) {
    Write-Host "  Found archive.zip – extracting and processing..." -ForegroundColor Green
    & $PY data\download_dataset.py
    Write-Host "  Dataset ready." -ForegroundColor Green
} elseif (Test-Path $kaggleCreds) {
    Write-Host "  Kaggle credentials found – downloading dataset..." -ForegroundColor Green
    & $PY data\download_dataset.py
    Write-Host "  Dataset ready." -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "  WARNING: No dataset found. Do ONE of the following:" -ForegroundColor Yellow
    Write-Host "    Option A (manual): Copy archive.zip to data\raw\archive.zip" -ForegroundColor Yellow
    Write-Host "    Option B (Kaggle): Add kaggle.json to $kaggleCreds" -ForegroundColor Yellow
    Write-Host "  Then re-run: .\bootstrap.ps1 -Train" -ForegroundColor Yellow
}

# ── 5. Copy trained model if present ─────────────────────────
Step "5/5  Checking for pre-trained model"

$modelSrc = "$ROOT\models\saved\stutter_model.pt"
if (Test-Path $modelSrc) {
    Write-Host "  Found stutter_model.pt – ready for inference." -ForegroundColor Green
} else {
    Write-Host "  No trained model found." -ForegroundColor Yellow
    Write-Host "  Train one with:"
    Write-Host "    .\.venv\Scripts\python.exe training\train.py --model cnn --mode audio_only --epochs 50 --batch 32"
}

# ── Optional training ─────────────────────────────────────────
if ($Train) {
    Step "BONUS  Training model (--model cnn, 50 epochs)"
    & $PY training\train.py --model cnn --mode audio_only --epochs 50 --batch 16 --workers 0 --csv data\processed\sep28k_labels.csv --checkpoint stutter_model
}

# ── Done ─────────────────────────────────────────────────────
Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Start the app:" -ForegroundColor White
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host "    python run.py"
Write-Host ""
Write-Host "  API  →  http://localhost:8000"
Write-Host "  UI   →  http://localhost:8501"
Write-Host ("=" * 60) -ForegroundColor Green
