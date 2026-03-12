"""
api/main.py
────────────
FastAPI REST backend for the Stutter Detection System.

Endpoints
─────────
POST /detect-stutter
  • Accepts a video file (multipart/form-data)
  • Returns JSON with prediction results

GET  /health       – liveness check
GET  /classes      – list of detectable stutter types

Run locally
───────────
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import uuid
import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import UPLOAD_DIR, STUTTER_CLASSES, MODELS_DIR
from inference.pipeline import StutterInferencePipeline, InferenceResult

# ─────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Stutter Detection API",
    description = "Multimodal ML system to detect stuttering in uploaded videos.",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ─────────────────────────────────────────────────────────────────────────
# Lazy-loaded pipeline (initialised on first request)
# ─────────────────────────────────────────────────────────────────────────

_pipeline: StutterInferencePipeline | None = None


def get_pipeline() -> StutterInferencePipeline:
    global _pipeline
    if _pipeline is None:
        checkpoint = str(MODELS_DIR / "stutter_model.pt")
        if not Path(checkpoint).exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model checkpoint not found. "
                    "Please train the model first: "
                    "python training/train.py"
                ),
            )
        _pipeline = StutterInferencePipeline(
            checkpoint_path=checkpoint,
            audio_only=True,
            use_cnn=True,        # Use SpectrogramCNN – matches the trained checkpoint
            # post‑processing thresholds (lowered to be less aggressive)
            min_confidence=0.35,  # Lower threshold for initial prediction
            min_stutter_pct=5.0,  # Lower threshold - only override if <5% stutter detected
        )
    return _pipeline


# ─────────────────────────────────────────────────────────────────────────
# Response schema
# ─────────────────────────────────────────────────────────────────────────

class DetectionResponse(BaseModel):
    predicted_label: str
    confidence:      float
    stutter_pct:     float
    duration_sec:    float
    class_probs:     dict
    timeline:        list


# ─────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "Stutter Detection API"}


@app.get("/classes")
def get_classes():
    return {"classes": STUTTER_CLASSES}


@app.post("/detect-stutter", response_model=DetectionResponse)
async def detect_stutter(file: UploadFile = File(...)):
    """
    Upload a video file and receive stuttering analysis results.

    Accepted formats: mp4, avi, mov, mkv, webm
    """
    # ── Validate file type ────────────────────────────────────────────
    allowed_ext = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    suffix      = Path(file.filename).suffix.lower()
    if suffix not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. "
                   f"Allowed: {', '.join(allowed_ext)}",
        )

    # ── Save uploaded file ────────────────────────────────────────────
    save_path = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
    try:
        with save_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save failed: {e}")
    finally:
        await file.close()

    # ── Run inference ─────────────────────────────────────────────────
    try:
        pipeline = get_pipeline()
        result: InferenceResult = pipeline.run(str(save_path))
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # ── Clean up uploaded file ────────────────────────────────────────
    save_path.unlink(missing_ok=True)

    return DetectionResponse(**result.to_dict())
