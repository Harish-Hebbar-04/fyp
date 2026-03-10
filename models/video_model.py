"""
models/video_model.py
──────────────────────
Video branch: encodes the (T, F) facial-landmark sequence into a
fixed-size embedding using a bidirectional GRU + attention pooling.

Two sub-architectures:

1. FacialGRU  – bidirectional GRU, 2-layer, with learnable attention
2. FacialMLP  – simple MLP on the stat-pooled vector (fast fallback)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import NUM_FACE_FEATURES, VIDEO_EMBED_DIM, DROPOUT, MAX_FRAMES


# ─────────────────────────────────────────────────────────────────────────
# 1. Bidirectional GRU with Attention
# ─────────────────────────────────────────────────────────────────────────

class FacialGRU(nn.Module):
    """
    Processes a (B, T, NUM_FACE_FEATURES) sequence.
    Uses a 2-layer BiGRU + single-head attention pooling.
    """

    def __init__(
        self,
        input_dim:  int = NUM_FACE_FEATURES,
        hidden_dim: int = 128,
        embed_dim:  int = VIDEO_EMBED_DIM,
        num_layers: int = 2,
        dropout:    float = DROPOUT,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        gru_out_dim = hidden_dim * 2   # bidirectional

        # Learnable attention over time steps
        self.attn_linear = nn.Linear(gru_out_dim, 1)

        self.projection = nn.Sequential(
            nn.Linear(gru_out_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, input_dim)

        Returns
        -------
        (B, embed_dim)
        """
        x   = self.input_proj(x)          # (B, T, hidden_dim)
        out, _ = self.gru(x)              # (B, T, gru_out_dim)

        # Attention weights
        scores  = self.attn_linear(out)   # (B, T, 1)
        weights = F.softmax(scores, dim=1)
        context = (weights * out).sum(dim=1)   # (B, gru_out_dim)

        return self.projection(context)   # (B, embed_dim)


# ─────────────────────────────────────────────────────────────────────────
# 2. Lightweight MLP on stat-pooled features
# ─────────────────────────────────────────────────────────────────────────

class FacialMLP(nn.Module):
    """
    Expects a pre-computed stat-pooled vector of shape
    (B, 4 * NUM_FACE_FEATURES * 2) ≈ (B, 80).
    """

    def __init__(
        self,
        input_dim: int = 4 * NUM_FACE_FEATURES * 2,   # 80
        embed_dim: int = VIDEO_EMBED_DIM,
        dropout:   float = DROPOUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, input_dim)

        Returns
        -------
        (B, embed_dim)
        """
        return self.net(x)
