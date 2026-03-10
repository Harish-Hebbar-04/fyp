"""
video_processing/feature_extraction.py
───────────────────────────────────────
Converts the raw (T, 10) facial landmark sequence into a fixed-size
feature vector for the video branch of the fusion model.

Strategy
────────
1. Temporal stats  – mean, std, max, min across time → 4 * 10 = 40 values
2. Delta stats      – first-order differences → 4 * 10 = 40 values
3. Optional: full sequence padded to (MAX_FRAMES, 10) for an LSTM branch.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MAX_FRAMES, NUM_FACE_FEATURES


def sequence_to_vector(seq: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    seq : np.ndarray of shape (T, NUM_FACE_FEATURES)

    Returns
    -------
    np.ndarray of shape (4 * NUM_FACE_FEATURES * 2,) ≈ 80-D
    """
    delta = np.diff(seq, axis=0)                         # (T-1, F)

    def stats(x: np.ndarray) -> np.ndarray:
        return np.concatenate([
            x.mean(axis=0),
            x.std(axis=0),
            x.max(axis=0),
            x.min(axis=0),
        ])

    feat = np.concatenate([stats(seq), stats(delta)])    # 80-D
    return feat.astype(np.float32)


def normalise_sequence(seq: np.ndarray) -> np.ndarray:
    """Z-normalise a (T, F) sequence along the time axis."""
    mu  = seq.mean(axis=0, keepdims=True)
    std = seq.std(axis=0,  keepdims=True) + 1e-8
    return ((seq - mu) / std).astype(np.float32)
