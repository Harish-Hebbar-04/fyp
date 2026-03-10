"""
Central configuration for the Stutter Detection System.
All hyperparameters, paths, and constants live here.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
DATA_DIR       = BASE_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
MODELS_DIR     = BASE_DIR / "models" / "saved"

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Stutter Labels
# ─────────────────────────────────────────────
STUTTER_CLASSES = ["Fluent", "Repetition", "Prolongation", "Block", "Interjection"]
NUM_CLASSES     = len(STUTTER_CLASSES)
LABEL2ID        = {label: idx for idx, label in enumerate(STUTTER_CLASSES)}
ID2LABEL        = {idx: label for label, idx in LABEL2ID.items()}

# SEP-28k column names (official CSV headers)
SEP28K_COLUMNS = [
    "Show", "EpId", "ClipId",
    "Prolongation", "Block", "SoundRep", "WordRep",
    "Interjection", "NoStutteredWords"
]

# ─────────────────────────────────────────────
# Audio settings
# ─────────────────────────────────────────────
SAMPLE_RATE        = 16_000   # Hz – required by Wav2Vec2
MAX_AUDIO_LENGTH   = 5        # seconds per clip
N_MFCC             = 40
N_MELS             = 128
HOP_LENGTH         = 512
N_FFT              = 2048
WAV2VEC2_MODEL     = "facebook/wav2vec2-base-960h"

# ─────────────────────────────────────────────
# Video settings
# ─────────────────────────────────────────────
FPS                = 30
MAX_FRAMES         = 150      # 5 s × 30 fps
NUM_FACE_FEATURES  = 20       # derived facial landmark features per frame

# ─────────────────────────────────────────────
# Model / Training
# ─────────────────────────────────────────────
AUDIO_EMBED_DIM    = 256
VIDEO_EMBED_DIM    = 128
FUSION_DIM         = 256
DROPOUT            = 0.2      # reduced to let smaller CNN generalise

BATCH_SIZE         = 32       # larger batch → more stable gradients
LEARNING_RATE      = 1e-3     # CNN trains from scratch – needs higher LR
NUM_EPOCHS         = 50       # sufficient with cosine warm-restarts
WEIGHT_DECAY       = 1e-3     # stronger L2 regularisation
PATIENCE           = 12       # early-stopping patience (on val macro-F1)

TRAIN_SPLIT        = 0.8
VAL_SPLIT          = 0.1
TEST_SPLIT         = 0.1

# ─────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────
AUGMENT_PROB       = 0.7      # higher prob – especially important for minority classes
NOISE_FACTOR       = 0.008
PITCH_SHIFT_STEPS  = 3        # semitones
TIME_STRETCH_RATE  = 0.85     # 0.85 – 1.15

# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────
API_HOST           = "0.0.0.0"
API_PORT           = 8000
UPLOAD_DIR         = BASE_DIR / "api" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Misc
# ─────────────────────────────────────────────
RANDOM_SEED        = 42
DEVICE             = "cuda"   # will fall back to cpu in code
