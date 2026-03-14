"""
training/dataset.py
────────────────────
PyTorch Dataset for the SEP-28k stutter detection task.

Modes
─────
• 'audio_only'  – returns (waveform_tensor, label)
• 'full'        – returns (waveform_tensor, landmark_vec_tensor, label)
  (landmark features must be pre-computed and stored in a .npy sidecar)

The dataset reads from the CSV produced by data/download_dataset.py:
    data/processed/sep28k_labels.csv
"""

import sys
import numpy as np
import pandas as pd
import torch
import librosa
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2Processor

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    PROCESSED_DIR, SAMPLE_RATE, MAX_AUDIO_LENGTH,
    NUM_CLASSES, LABEL2ID, ID2LABEL, WAV2VEC2_MODEL,
    BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED,
    NUM_FACE_FEATURES, MAX_FRAMES, N_MELS, N_FFT, HOP_LENGTH,
)
from audio_processing.feature_extraction import load_audio
from training.augmentation import augment_waveform, add_gaussian_noise, random_gain, spec_augment


# ─────────────────────────────────────────────────────────────────────────
# Core Dataset
# ─────────────────────────────────────────────────────────────────────────

class StutterDataset(Dataset):
    """
    Parameters
    ----------
    df         : DataFrame with columns [audio_path, label, label_id]
    processor  : Wav2Vec2Processor for tokenising waveforms
    augment    : Whether to apply data augmentation
    mode       : 'audio_only' | 'full'
    """

    def __init__(
        self,
        df: pd.DataFrame,
        processor: Wav2Vec2Processor,
        augment:  bool = False,
        mode:     str  = "audio_only",
        cnn_mode: bool = False,
    ):
        self.df        = df.reset_index(drop=True)
        self.processor = processor
        self.augment   = augment
        self.mode      = mode
        self.cnn_mode  = cnn_mode
        self.max_len   = SAMPLE_RATE * MAX_AUDIO_LENGTH
        # Two-tier in-memory cache: raw waveform + log-Mel
        # Populated on first access; reused across every subsequent epoch.
        self._wav_cache: dict = {}
        self._mel_cache: dict = {}

    def __len__(self):
        return len(self.df)

    def _load_wav(self, idx: int) -> np.ndarray:
        """Load and cache the raw waveform for sample idx."""
        if idx not in self._wav_cache:
            self._wav_cache[idx] = load_audio(self.df.iloc[idx]["audio_path"])
        return self._wav_cache[idx]

    def _load_mel(self, idx: int) -> np.ndarray:
        """Compute and cache the log-Mel spectrogram for sample idx."""
        if idx not in self._mel_cache:
            wav = self._load_wav(idx)
            mel = librosa.feature.melspectrogram(
                y=wav, sr=SAMPLE_RATE,
                n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
            )
            self._mel_cache[idx] = librosa.power_to_db(
                mel, ref=np.max
            ).astype(np.float32)
        return self._mel_cache[idx]

    def __getitem__(self, idx: int) -> dict:
        label = torch.tensor(
            int(self.df.iloc[idx]["label_id"]), dtype=torch.long
        )

        # ── CNN mode (fast path) ────────────────────────────────────────
        # Always serves the cached log-Mel – no per-sample recomputation.
        # Wav2Vec2Processor is completely skipped.
        # MixUp + SpecAugment in the Trainer handle all augmentation variance.
        if self.cnn_mode:
            mel_tensor = torch.from_numpy(self._load_mel(idx)).unsqueeze(0)
            return {"mel_spec": mel_tensor,
                    "input_values": torch.zeros(1),   # dummy – not used by CNN
                    "label": label}

        # ── Wav2Vec2 mode (full path) ───────────────────────────────────
        waveform = self._load_wav(idx).copy()
        if self.augment:
            waveform = augment_waveform(waveform)

        inputs = self.processor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
        )
        audio_tensor = inputs.input_values.squeeze(0)   # (max_len,)
        mel_tensor   = torch.from_numpy(self._load_mel(idx)).unsqueeze(0)
        item = {"input_values": audio_tensor, "mel_spec": mel_tensor,
                "label": label}

        # ── Facial landmarks (full mode) ────────────────────────────────
        if self.mode == "full":
            npy_path = Path(self.df.iloc[idx]["audio_path"]).with_suffix(".npy")
            if npy_path.exists():
                lm_seq = np.load(str(npy_path))
            else:
                lm_seq = np.zeros((MAX_FRAMES, NUM_FACE_FEATURES), dtype=np.float32)
            if len(lm_seq) < MAX_FRAMES:
                pad    = np.zeros((MAX_FRAMES - len(lm_seq), NUM_FACE_FEATURES),
                                  dtype=np.float32)
                lm_seq = np.vstack([lm_seq, pad])
            else:
                lm_seq = lm_seq[:MAX_FRAMES]
            item["landmark_seq"] = torch.tensor(lm_seq, dtype=torch.float32)
        return item


# ─────────────────────────────────────────────────────────────────────────
# Helper: class-balanced WeightedRandomSampler
# ─────────────────────────────────────────────────────────────────────────

def make_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Return a WeightedRandomSampler that aggressively oversamples minority
    classes by using squared-inverse-frequency weights.

    This gives Interjection / Prolongation clips ~15× more draw probability
    than Fluent clips, so the model sees a roughly balanced batch distribution
    even though the raw dataset is highly skewed.
    """
    class_counts  = df["label_id"].value_counts().sort_index().values
    # Squared inverse-frequency — much stronger emphasis on minority classes
    class_weights = (1.0 / class_counts) ** 1.5
    class_weights /= class_weights.sum()                   # normalise
    sample_weights = class_weights[df["label_id"].values]

    # Oversample to 2× the dataset size so every minority sample is replayed
    num_samples = len(sample_weights) * 2

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights.astype("float32")),
        num_samples=num_samples,
        replacement=True,
    )


# ─────────────────────────────────────────────────────────────────────────
# Factory: build train / val / test DataLoaders
# ─────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    csv_path: str | None = None,
    batch_size: int       = BATCH_SIZE,
    mode: str             = "audio_only",
    num_workers: int      = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).

    Splits are stratified by label.
    Train loader uses WeightedRandomSampler for class balance.
    """
    if csv_path is None:
        csv_path = PROCESSED_DIR / "sep28k_labels.csv"

    df = pd.read_csv(csv_path)
    processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL)

    # Stratified split
    train_df, temp_df = train_test_split(
        df, test_size=1 - TRAIN_SPLIT,
        stratify=df["label_id"], random_state=RANDOM_SEED,
    )
    val_ratio = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - val_ratio,
        stratify=temp_df["label_id"], random_state=RANDOM_SEED,
    )

    print(f"Dataset splits →  train:{len(train_df)}  "
          f"val:{len(val_df)}  test:{len(test_df)}")

    train_ds = StutterDataset(train_df, processor, augment=True,  mode=mode, cnn_mode=True)
    val_ds   = StutterDataset(val_df,   processor, augment=False, mode=mode, cnn_mode=True)
    test_ds  = StutterDataset(test_df,  processor, augment=False, mode=mode, cnn_mode=True)

    sampler = make_weighted_sampler(train_df)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader
