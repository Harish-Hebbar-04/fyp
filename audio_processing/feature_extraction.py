"""
audio_processing/feature_extraction.py
───────────────────────────────────────
Extracts both hand-crafted (MFCC, Mel-spectrogram, pitch, energy)
and deep (Wav2Vec2) features from an audio waveform.

All functions expect a 1-D NumPy array at SAMPLE_RATE Hz.
"""

import sys
import numpy as np
import torch
import librosa
from pathlib import Path
from transformers import Wav2Vec2Processor, Wav2Vec2Model

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    SAMPLE_RATE, N_MFCC, N_MELS, HOP_LENGTH, N_FFT,
    MAX_AUDIO_LENGTH, WAV2VEC2_MODEL,
)

# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def load_audio(path: str) -> np.ndarray:
    """Load audio, resample to SAMPLE_RATE, and pad / trim to MAX_AUDIO_LENGTH."""
    waveform, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    target_len = SAMPLE_RATE * MAX_AUDIO_LENGTH
    if len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    else:
        waveform = waveform[:target_len]
    return waveform.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# Hand-crafted features
# ─────────────────────────────────────────────────────────────────────────

def extract_mfcc(waveform: np.ndarray) -> np.ndarray:
    """
    Returns MFCC + delta + delta-delta of shape (3*N_MFCC, T).
    Each column is one time frame.
    """
    mfcc  = librosa.feature.mfcc(y=waveform, sr=SAMPLE_RATE,
                                  n_mfcc=N_MFCC, n_fft=N_FFT,
                                  hop_length=HOP_LENGTH)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2])   # (120, T)


def extract_mel_spectrogram(waveform: np.ndarray) -> np.ndarray:
    """Returns log-Mel spectrogram of shape (N_MELS, T)."""
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=SAMPLE_RATE,
        n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    return librosa.power_to_db(mel, ref=np.max)


def extract_pitch(waveform: np.ndarray) -> np.ndarray:
    """
    Fundamental frequency (F0) using PYIN.
    Returns a 1-D array of shape (T,) in Hz (NaN where unvoiced).
    """
    f0, _, _ = librosa.pyin(
        waveform, fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=SAMPLE_RATE, hop_length=HOP_LENGTH
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    return f0.astype(np.float32)


def extract_energy(waveform: np.ndarray) -> np.ndarray:
    """RMS energy per frame, shape (T,)."""
    rms = librosa.feature.rms(y=waveform, frame_length=N_FFT,
                               hop_length=HOP_LENGTH)[0]
    return rms.astype(np.float32)


def extract_handcrafted_features(waveform: np.ndarray) -> np.ndarray:
    """
    Combine MFCC, Mel, pitch, and energy into a fixed-length feature vector
    by taking the mean and std across time for each feature map.

    Returns a 1-D vector of shape (D,).
    """
    mfcc    = extract_mfcc(waveform)          # (120, T)
    mel     = extract_mel_spectrogram(waveform)  # (128, T)
    pitch   = extract_pitch(waveform)         # (T,)
    energy  = extract_energy(waveform)        # (T,)

    def stat_pool(x):
        """Mean + std pooling → 2*channels."""
        return np.concatenate([x.mean(axis=-1), x.std(axis=-1)])

    feat = np.concatenate([
        stat_pool(mfcc),            # 240
        stat_pool(mel),             # 256
        np.array([pitch.mean(), pitch.std(), pitch.max()]),   # 3
        np.array([energy.mean(), energy.std(), energy.max()]),  # 3
    ])
    return feat.astype(np.float32)   # ~ 502-D


# ─────────────────────────────────────────────────────────────────────────
# Deep (Wav2Vec2) features
# ─────────────────────────────────────────────────────────────────────────

class Wav2Vec2FeatureExtractor:
    """
    Wraps facebook/wav2vec2-base-960h and returns mean-pooled hidden states.

    Usage:
        extractor = Wav2Vec2FeatureExtractor()
        emb = extractor(waveform)   # np.ndarray shape (768,)
    """

    def __init__(self, model_name: str = WAV2VEC2_MODEL, device: str = "cpu"):
        self.device    = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model     = Wav2Vec2Model.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        waveform : float32 array at 16 kHz, length = SAMPLE_RATE * secs

        Returns
        -------
        np.ndarray of shape (hidden_size,) – typically 768
        """
        inputs = self.processor(
            waveform, sampling_rate=SAMPLE_RATE,
            return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self.device)

        outputs = self.model(input_values)
        # Mean-pool over time dimension
        hidden = outputs.last_hidden_state  # (1, T, H)
        embedding = hidden.mean(dim=1).squeeze(0)  # (H,)
        return embedding.cpu().numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# Combined extractor
# ─────────────────────────────────────────────────────────────────────────

def extract_all_audio_features(
    waveform: np.ndarray,
    wav2vec_extractor: Wav2Vec2FeatureExtractor | None = None,
) -> dict:
    """
    Returns a dict with keys:
        'handcrafted'  – np.ndarray (~502-D)
        'wav2vec2'     – np.ndarray (768-D), only if extractor provided
        'mfcc_2d'      – np.ndarray (120, T) for CNN input
        'mel_2d'       – np.ndarray (128, T) for CNN input
    """
    result = {
        "handcrafted": extract_handcrafted_features(waveform),
        "mfcc_2d":     extract_mfcc(waveform),
        "mel_2d":      extract_mel_spectrogram(waveform),
    }
    if wav2vec_extractor is not None:
        result["wav2vec2"] = wav2vec_extractor(waveform)
    return result
