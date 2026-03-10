"""
models/audio_model.py
──────────────────────
Audio branch of the multimodal stutter detection system.

Two sub-architectures are available:

1. Wav2Vec2Branch  – fine-tunes facebook/wav2vec2-base-960h
   • The encoder is partially frozen (bottom N layers frozen)
   • A projection head compresses the 768-D CLS/mean-pool to AUDIO_EMBED_DIM

2. SpectrogramCNN  – lightweight CNN on log-Mel spectrograms
   • Used as a faster fallback when Wav2Vec2 is too slow
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from transformers import Wav2Vec2Model

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    WAV2VEC2_MODEL, AUDIO_EMBED_DIM, DROPOUT,
    N_MELS, NUM_CLASSES,
)


# ─────────────────────────────────────────────────────────────────────────
# 1. Wav2Vec2 Branch
# ─────────────────────────────────────────────────────────────────────────

class Wav2Vec2Branch(nn.Module):
    """
    Fine-tunable Wav2Vec2 encoder → mean-pool → projection head.

    Parameters
    ----------
    freeze_layers : int
        Number of bottom transformer layers to keep frozen.
        Set to 0 to fine-tune all layers (slower but better).
    """

    def __init__(
        self,
        model_name: str = WAV2VEC2_MODEL,
        embed_dim: int  = AUDIO_EMBED_DIM,
        dropout: float  = DROPOUT,
        freeze_layers: int = 6,
    ):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)

        # Freeze feature-extractor (CNN) – always
        for p in self.wav2vec2.feature_extractor.parameters():
            p.requires_grad = False

        # Optionally freeze bottom transformer layers
        for i, layer in enumerate(self.wav2vec2.encoder.layers):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False

        hidden_size = self.wav2vec2.config.hidden_size   # 768

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, input_values: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        input_values  : (B, T) float32 waveform at 16 kHz
        attention_mask: (B, T) optional

        Returns
        -------
        torch.Tensor of shape (B, embed_dim)
        """
        outputs = self.wav2vec2(input_values=input_values,
                                attention_mask=attention_mask)
        hidden  = outputs.last_hidden_state          # (B, T', 768)
        pooled  = hidden.mean(dim=1)                  # (B, 768)
        return self.projection(pooled)               # (B, embed_dim)


# ─────────────────────────────────────────────────────────────────────────
# 2. Spectrogram CNN Branch (improved: deeper + Squeeze-and-Excitation)
# ─────────────────────────────────────────────────────────────────────────

class _SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class _CNNBlock(nn.Module):
    """Double-conv block with BN, ReLU, SE-attention and MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.se   = _SEBlock(out_ch)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = self.se(x)
        x = self.pool(x)
        return self.drop(x)


class SpectrogramCNN(nn.Module):
    """
    Deeper CNN with Squeeze-and-Excitation attention that operates on
    log-Mel spectrograms of shape (B, 1, N_MELS, T).

    Architecture: 4 × _CNNBlock (32→64→128→256) + AdaptiveAvgPool → MLP head
    SE blocks focus the network on discriminative frequency bands, which
    helps distinguish stutter types (e.g. Prolongation vs Block).
    """

    def __init__(self, embed_dim: int = AUDIO_EMBED_DIM, dropout: float = DROPOUT):
        super().__init__()

        self.features = nn.Sequential(
            _CNNBlock(1,   32,  dropout=0.05),
            _CNNBlock(32,  64,  dropout=0.05),
            _CNNBlock(64,  128, dropout=0.10),
            _CNNBlock(128, 256, dropout=0.10),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        spectrogram : (B, 1, N_MELS, T)

        Returns
        -------
        (B, embed_dim)
        """
        x = self.features(spectrogram)  # (B, 256, H', W')
        x = self.pool(x)                # (B, 256, 1, 1)
        return self.head(x)             # (B, embed_dim)
