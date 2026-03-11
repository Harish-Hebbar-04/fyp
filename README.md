# 🎙️ Stutter Detection System

A **multimodal end-to-end machine learning system** that detects stuttering in speech from video input.  
It fine-tunes **Wav2Vec2** on the SEP-28k dataset and fuses audio embeddings with **MediaPipe facial landmark** features.

---

## Architecture

```
Video Upload
    │
    ├─► FFmpeg ──────────────► Mono 16-kHz WAV
    │           Audio Branch:
    │           Wav2Vec2 encoder → mean-pool → Linear → 256-D embedding
    │
    └─► OpenCV frames
         MediaPipe FaceMesh
         10 facial features / frame → BiGRU + attention → 128-D embedding
    
    Audio (256-D) ⊕ Video (128-D) → Fusion MLP → 5-class classifier
    
    Output: label | confidence | stutter % | timeline
```

---

## Detected Stutter Types

| Label          | Description                                      |
|----------------|--------------------------------------------------|
| **Fluent**     | Normal, uninterrupted speech                     |
| **Repetition** | Sound / syllable / word repetitions (e-e-eight)  |
| **Prolongation**| Extended sound durations (ssssix)               |
| **Block**      | Silent pauses mid-word (no airflow)              |
| **Interjection**| Filler words (um, uh, like)                     |

---

## Quick Start

### 1. Install Dependencies

```bash
# Create & activate venv (Windows)
python -m venv .venv
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```

> **FFmpeg required** — download from https://ffmpeg.org/download.html and add to PATH.

### 2. Download Dataset

```bash
# Configure Kaggle credentials first (see below)
python data/download_dataset.py
```

#### Kaggle Setup
1. Go to https://www.kaggle.com/settings → **API** → **Create New Token**
2. Save `kaggle.json` to `%USERPROFILE%\.kaggle\kaggle.json`

### 3. Train the Model

```bash
# Audio-only training (recommended starting point)
python training/train.py --mode audio_only --epochs 20

# Full multimodal training
python training/train.py --mode full --epochs 20

# Faster CNN variant (no Wav2Vec2)
python training/train.py --model cnn --mode audio_only
```

> **Tip:** if the trained model is overly eager to label fluent clips as
> stutter, reduce the class‑weight exponent in `config.py`
> (`CLASS_WEIGHT_EXP=1.0` is inverse‑frequency; higher values emphasise
> minorities more strongly).  You can also adjust the inference thresholds
> (`min_confidence`, `min_stutter_pct`) when constructing
> :class:`StutterInferencePipeline` (defaults already applied in the API/UI).

### 4. Run the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs: http://localhost:8000/docs

### 5. Run the UI

```bash
streamlit run ui/app.py
```

UI: http://localhost:8501

---

## Project Structure

```
Stuttering_detection/
├── config.py                    ← Central config (paths, hyperparameters)
├── requirements.txt
├── Dockerfile
│
├── data/
│   ├── raw/                     ← Downloaded SEP-28k files
│   ├── processed/               ← sep28k_labels.csv
│   └── download_dataset.py      ← Kaggle download + preprocessing
│
├── audio_processing/
│   ├── extract_audio.py         ← FFmpeg video→WAV extraction
│   └── feature_extraction.py   ← MFCC, Mel, Pitch, Energy, Wav2Vec2
│
├── video_processing/
│   ├── facial_landmarks.py      ← MediaPipe FaceMesh → (T,10) sequence
│   └── feature_extraction.py   ← Stat-pooling → fixed vector
│
├── models/
│   ├── audio_model.py           ← Wav2Vec2Branch + SpectrogramCNN
│   ├── video_model.py           ← FacialGRU + FacialMLP
│   ├── fusion_model.py          ← StutterDetectionModel (full)
│   └── saved/                   ← Trained checkpoints (.pt)
│
├── training/
│   ├── dataset.py               ← StutterDataset + DataLoaders
│   ├── augmentation.py          ← Noise, time-stretch, pitch-shift, SpecAugment
│   ├── trainer.py               ← Training loop, early stopping, metrics
│   └── train.py                 ← CLI entry-point
│
├── inference/
│   └── pipeline.py              ← StutterInferencePipeline (sliding window)
│
├── api/
│   └── main.py                  ← FastAPI: POST /detect-stutter
│
└── ui/
    └── app.py                   ← Streamlit dashboard
```

---

## API Reference

### `POST /detect-stutter`

**Request:** `multipart/form-data`  
- `file`: video file (mp4, avi, mov, mkv, webm)

**Response:**
```json
{
  "predicted_label": "Repetition",
  "confidence": 0.847,
  "stutter_pct": 62.5,
  "duration_sec": 12.3,
  "class_probs": {
    "Fluent": 0.083,
    "Repetition": 0.847,
    "Prolongation": 0.031,
    "Block": 0.022,
    "Interjection": 0.017
  },
  "timeline": [
    {"start_sec": 0.0, "end_sec": 5.0, "label": "Repetition", "confidence": 0.91, "probs": {...}},
    {"start_sec": 2.5, "end_sec": 7.5, "label": "Fluent",     "confidence": 0.76, "probs": {...}}
  ]
}
```

---

## Docker

```bash
# Build
docker build -t stutter-detection .

# Run (API + UI together)
docker run -p 8000:8000 -p 8501:8501 stutter-detection

# API only
docker run -p 8000:8000 stutter-detection uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## Tech Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| Speech model | Wav2Vec2 (HuggingFace) | SOTA self-supervised speech representations |
| Audio features | librosa, torchaudio | MFCC, Mel, pitch, energy extraction |
| Video features | MediaPipe FaceMesh | Accurate 468-point facial landmark tracking |
| Model | PyTorch | Flexible deep learning framework |
| API | FastAPI | Fast async REST API with auto-docs |
| UI | Streamlit + Plotly | Rapid ML dashboard with interactive charts |
| Audio extraction | FFmpeg | Universal video→audio conversion |
| Dataset | SEP-28k (Kaggle) | 28k labelled stuttering audio clips |

---

## Training Tips

- Start with `--mode audio_only` — SEP-28k is audio-only; video landmarks are available only for your own recordings.
- Use `--freeze 6` to keep the bottom 6 Wav2Vec2 layers frozen (speeds up training ~3×).
- Class imbalance is handled by `WeightedRandomSampler` + `CrossEntropyLoss(weight=...)`.
- SpecAugment + noise injection + pitch-shift prevent overfitting on small audio datasets.
- Early stopping (patience=5) prevents over-training; best model is auto-saved.
