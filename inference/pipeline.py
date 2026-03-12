"""
inference/pipeline.py
──────────────────────
End-to-end inference pipeline.

Flow
────
  Video File
      │
      ├─► FFmpeg ──► mono 16-kHz WAV
      │       │
      │       └─► Wav2Vec2Processor ──► audio tensor
      │
      ├─► OpenCV frames ──► MediaPipe FaceMesh ──► landmark sequence
      │
      └─► StutterDetectionModel ──► logits ──► probabilities
                                          │
                                          ▼
                              InferenceResult  (dataclass)
                              ├── predicted_label  str
                              ├── confidence        float
                              ├── class_probs       dict[str, float]
                              ├── stutter_pct       float
                              └── timeline          list[dict]

The timeline is produced by running inference on consecutive
overlapping 5-second windows.
"""

import sys
import tempfile
import numpy as np
import librosa as _librosa
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from pathlib import Path
from transformers import Wav2Vec2Processor

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    SAMPLE_RATE, MAX_AUDIO_LENGTH, WAV2VEC2_MODEL,
    MODELS_DIR, STUTTER_CLASSES, ID2LABEL, LABEL2ID,
    NUM_FACE_FEATURES, MAX_FRAMES, DEVICE,
    N_MELS, N_FFT, HOP_LENGTH,
)
from audio_processing.extract_audio      import extract_audio_ffmpeg, get_audio_duration
from audio_processing.feature_extraction import load_audio
from video_processing.facial_landmarks   import FacialLandmarkExtractor
from models.fusion_model                 import StutterDetectionModel, StutterDetectionCNN


# ─────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    predicted_label: str
    confidence:      float
    class_probs:     dict
    stutter_pct:     float
    timeline:        list = field(default_factory=list)
    duration_sec:    float = 0.0

    def to_dict(self) -> dict:
        return {
            "predicted_label": self.predicted_label,
            "confidence":      round(self.confidence, 4),
            "class_probs":     {k: round(v, 4) for k, v in self.class_probs.items()},
            "stutter_pct":     round(self.stutter_pct, 2),
            "timeline":        self.timeline,
            "duration_sec":    round(self.duration_sec, 2),
        }


# ─────────────────────────────────────────────────────────────────────────
# Inference Pipeline
# ─────────────────────────────────────────────────────────────────────────

class StutterInferencePipeline:
    """
    Loads a trained StutterDetectionModel and runs inference on a video file.

    Parameters
    ----------
    checkpoint_path : str | None
        Path to the saved .pt model weights.
        Defaults to models/saved/stutter_model.pt
    device : str
        'cuda' or 'cpu' (auto-detected if None)
    window_sec : float
        Duration of each sliding window for the timeline (default 5 s).
    stride_sec : float
        Step between consecutive windows (default 2.5 s → 50 % overlap).
    audio_only : bool
        Skip facial landmark extraction and run audio-only branch.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device:          str | None = None,
        window_sec:      float = MAX_AUDIO_LENGTH,
        stride_sec:      float = 2.5,
        audio_only:      bool  = True,
        use_cnn:         bool  = False,
        *,
        # post‑processing thresholds (see run() comments)
        min_confidence:   float = 0.5,
        min_stutter_pct:  float = 10.0,
    ):
        # ── Device ───────────────────────────────────────────────────────
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.use_cnn = use_cnn

        # ── Model ─────────────────────────────────────────────────────────
        if checkpoint_path is None:
            checkpoint_path = str(MODELS_DIR / "stutter_model.pt")

        if use_cnn:
            self.model = StutterDetectionCNN()
        else:
            self.model = StutterDetectionModel(audio_only=audio_only)

        state = torch.load(checkpoint_path, map_location=self.device,
                           weights_only=False)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        print(f"[Pipeline] {'CNN' if use_cnn else 'Wav2Vec2'} model loaded "
              f"from {checkpoint_path} on {self.device}")

        # ── Processor (Wav2Vec2 path only) ─────────────────────────────────
        if not use_cnn:
            self.processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL)

        # ── Mel transform (CNN path only) ──────────────────────────────────
        # Uses librosa (no torchaudio native dependency required)

        # ── FaceLandmark extractor ─────────────────────────────────────────
        self.audio_only = audio_only
        if not audio_only:
            self.face_extractor = FacialLandmarkExtractor()

        # thresholds for post‑processing (set by constructor args)
        self.min_confidence  = min_confidence
        self.min_stutter_pct = min_stutter_pct

        self.window_samples = int(window_sec * SAMPLE_RATE)
        self.stride_samples = int(stride_sec * SAMPLE_RATE)

    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _infer_window(
        self,
        waveform: np.ndarray,
        landmark_seq: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run model on one window; returns softmax probabilities (NUM_CLASSES,)."""

        # Pad / trim to window length
        w = waveform
        if len(w) < self.window_samples:
            w = np.pad(w, (0, self.window_samples - len(w)))
        w = w[:self.window_samples]

        if self.use_cnn:
            # CNN path: compute log-Mel with librosa (no torchaudio required)
            mel = _librosa.feature.melspectrogram(
                y=w, sr=SAMPLE_RATE, n_mels=N_MELS,
                n_fft=N_FFT, hop_length=HOP_LENGTH
            )
            mel_db = _librosa.power_to_db(mel, ref=np.max)
            mel_t = torch.tensor(mel_db, dtype=torch.float32,
                                 device=self.device).unsqueeze(0).unsqueeze(0)  # (1,1,N_MELS,T')
            lm_vec = torch.zeros(1, 4 * NUM_FACE_FEATURES * 2,
                                 device=self.device)  # dummy 80-D
            logits = self.model(mel_t, lm_vec)   # (1, NUM_CLASSES)
        else:
            inputs = self.processor(
                w, sampling_rate=SAMPLE_RATE,
                return_tensors="pt", padding=False,
            )
            input_values = inputs.input_values.to(self.device)   # (1, T)

            if self.audio_only or landmark_seq is None:
                lm = torch.zeros(1, MAX_FRAMES, NUM_FACE_FEATURES,
                                 device=self.device)
            else:
                lm_arr = landmark_seq
                if len(lm_arr) < MAX_FRAMES:
                    pad = np.zeros(
                        (MAX_FRAMES - len(lm_arr), NUM_FACE_FEATURES), dtype=np.float32
                    )
                    lm_arr = np.vstack([lm_arr, pad])
                lm = torch.tensor(lm_arr[:MAX_FRAMES], dtype=torch.float32
                                   ).unsqueeze(0).to(self.device)

            logits = self.model(input_values, lm)    # (1, NUM_CLASSES)

        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs

    # ─────────────────────────────────────────────────────────────────────

    def run(self, video_path: str) -> InferenceResult:
        """
        Full pipeline:  video → extract audio (+ optionally landmarks)
                       → sliding-window inference → aggregate result.
        """
        video_path = str(video_path)

        # ── 1. Extract audio ──────────────────────────────────────────────
        wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_tmp.close()
        audio_path = extract_audio_ffmpeg(video_path, wav_tmp.name)
        waveform   = load_audio(audio_path)
        duration   = get_audio_duration(audio_path)

        # ── 2. Extract landmarks (optional) ──────────────────────────────
        landmark_seq = None
        if not self.audio_only:
            landmark_seq = self.face_extractor(video_path)  # (T, F)

        # ── 3. Sliding-window inference ───────────────────────────────────
        timeline = []
        start    = 0
        while start < len(waveform):
            end   = start + self.window_samples
            chunk = waveform[start:end]

            lm_chunk = None
            if landmark_seq is not None:
                fps_start = int((start / SAMPLE_RATE) * MAX_FRAMES / MAX_AUDIO_LENGTH)
                fps_end   = fps_start + MAX_FRAMES
                lm_chunk  = landmark_seq[fps_start:fps_end]

            probs = self._infer_window(chunk, lm_chunk)
            pred_id    = int(np.argmax(probs))
            pred_label = ID2LABEL[pred_id]

            timeline.append({
                "start_sec":  round(start / SAMPLE_RATE, 2),
                "end_sec":    round(min(end, len(waveform)) / SAMPLE_RATE, 2),
                "label":      pred_label,
                "confidence": round(float(probs[pred_id]), 4),
                "probs":      {STUTTER_CLASSES[i]: round(float(p), 4)
                               for i, p in enumerate(probs)},
            })

            start += self.stride_samples

        # ── 4. Aggregate ──────────────────────────────────────────────────
        all_probs   = np.array([e["probs"][c]
                                for e in timeline
                                for c in STUTTER_CLASSES
                                ]).reshape(len(timeline), -1)
        mean_probs  = all_probs.mean(axis=0)
        pred_id     = int(np.argmax(mean_probs))
        pred_label  = ID2LABEL[pred_id]
        confidence  = float(mean_probs[pred_id])

        fluent_id   = LABEL2ID["Fluent"]
        # Calculate stutter_pct as 100 - (mean probability of Fluent class)
        # This is more robust than counting hard predictions
        stutter_pct = float((1.0 - mean_probs[fluent_id]) * 100)

        # ── Post-processing to reduce false positives ────────────────────
        # If the model votes for a stutter class but the confidence is low
        # and only a small portion of windows are non‑fluent, override to
        # fluent.  Thresholds are configurable via attributes set in
        # __init__ (defaults chosen heuristically).
        if pred_id != fluent_id:
            if confidence < self.min_confidence:
                pred_id = fluent_id
                pred_label = "Fluent"
                confidence = float(mean_probs[fluent_id])
                stutter_pct = 0.0  # Reset stutter_pct when overriding to Fluent
            elif stutter_pct < self.min_stutter_pct:
                pred_id = fluent_id
                pred_label = "Fluent"
                confidence = float(mean_probs[fluent_id])
                stutter_pct = 0.0  # Reset stutter_pct when overriding to Fluent
        
        # Additional safeguard: if Fluent probability is high, always prefer Fluent
        # even if a stutter class has slightly higher probability
        if mean_probs[fluent_id] > 0.45 and pred_id != fluent_id:
            pred_id = fluent_id
            pred_label = "Fluent"
            confidence = float(mean_probs[fluent_id])
            stutter_pct = float((1.0 - mean_probs[fluent_id]) * 100)

        return InferenceResult(
            predicted_label = pred_label,
            confidence      = confidence,
            class_probs     = {STUTTER_CLASSES[i]: float(mean_probs[i])
                               for i in range(len(STUTTER_CLASSES))},
            stutter_pct     = stutter_pct,
            timeline        = timeline,
            duration_sec    = duration,
        )
