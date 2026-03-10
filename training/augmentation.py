"""
training/augmentation.py
─────────────────────────
On-the-fly audio augmentation applied during training.

Augmentations
─────────────
• GaussianNoise   – adds white noise
• TimeStretch     – speeds up / slows down without changing pitch
• PitchShift      – shifts pitch without changing speed
• RandomGain      – random amplitude scaling
• SpecAugment     – frequency & time masking on log-Mel spectrograms
"""

import sys
import random
import numpy as np
import librosa
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    AUGMENT_PROB, NOISE_FACTOR,
    PITCH_SHIFT_STEPS, TIME_STRETCH_RATE, SAMPLE_RATE,
)


def add_gaussian_noise(waveform: np.ndarray,
                        noise_factor: float = NOISE_FACTOR) -> np.ndarray:
    noise = np.random.randn(len(waveform)).astype(np.float32)
    return waveform + noise_factor * noise


def time_stretch(waveform: np.ndarray,
                 rate_range: tuple = (TIME_STRETCH_RATE, 1.1)) -> np.ndarray:
    rate = random.uniform(*rate_range)
    return librosa.effects.time_stretch(waveform, rate=rate)


def pitch_shift(waveform: np.ndarray,
                steps_range: int = PITCH_SHIFT_STEPS) -> np.ndarray:
    n_steps = random.uniform(-steps_range, steps_range)
    return librosa.effects.pitch_shift(
        waveform, sr=SAMPLE_RATE, n_steps=n_steps
    )


def random_gain(waveform: np.ndarray,
                gain_range: tuple = (0.7, 1.3)) -> np.ndarray:
    gain = random.uniform(*gain_range)
    return (waveform * gain).clip(-1.0, 1.0)


def augment_waveform(waveform: np.ndarray,
                      prob: float = AUGMENT_PROB) -> np.ndarray:
    """
    Randomly apply a chain of augmentations.

    Each augmentation is applied independently with probability `prob`.
    The output has the SAME length as input (padded / trimmed if needed).
    """
    original_len = len(waveform)

    if random.random() < prob:
        waveform = add_gaussian_noise(waveform)
    if random.random() < prob:
        waveform = time_stretch(waveform)
    if random.random() < prob:
        waveform = pitch_shift(waveform)
    if random.random() < prob:
        waveform = random_gain(waveform)

    # Ensure same length
    if len(waveform) < original_len:
        waveform = np.pad(waveform, (0, original_len - len(waveform)))
    else:
        waveform = waveform[:original_len]

    return waveform.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# SpecAugment (applied to log-Mel tensors / numpy arrays)
# ─────────────────────────────────────────────────────────────────────────

def spec_augment(
    mel: np.ndarray,
    freq_mask_param: int = 15,
    time_mask_param: int = 35,
    num_freq_masks:  int = 2,
    num_time_masks:  int = 2,
) -> np.ndarray:
    """
    Apply SpecAugment to a (freq_bins, time_steps) log-Mel spectrogram.

    Source: Park et al. 2019 – https://arxiv.org/abs/1904.08779
    """
    mel = mel.copy()
    F, T = mel.shape

    # Frequency masking
    for _ in range(num_freq_masks):
        f  = random.randint(0, freq_mask_param)
        f0 = random.randint(0, F - f)
        mel[f0: f0 + f, :] = mel.mean()

    # Time masking
    for _ in range(num_time_masks):
        t  = random.randint(0, min(time_mask_param, T))
        t0 = random.randint(0, T - t)
        mel[:, t0: t0 + t] = mel.mean()

    return mel
