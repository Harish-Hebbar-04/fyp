"""
models/fusion_model.py
───────────────────────
Multimodal Fusion Model for Stutter Detection.

Architecture
────────────
┌──────────────┐    ┌──────────────┐
│  Audio Branch│    │  Video Branch│
│ Wav2Vec2     │    │ FacialGRU    │
│ (256-D emb)  │    │ (128-D emb)  │
└──────┬───────┘    └──────┬───────┘
       │                   │
       └────────┬──────────┘
          Concatenate (384-D)
                │
         Fusion MLP
       ┌────────────────┐
       │  Linear(384,256)│
       │  GELU, Dropout │
       │  Linear(256,128)│
       │  GELU, Dropout │
       └────────────────┘
                │
       Linear(128, NUM_CLASSES)
                │
         Softmax (inference)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    AUDIO_EMBED_DIM, VIDEO_EMBED_DIM, FUSION_DIM,
    NUM_CLASSES, DROPOUT,
)
from models.audio_model import Wav2Vec2Branch, SpectrogramCNN
from models.video_model  import FacialGRU, FacialMLP


# ─────────────────────────────────────────────────────────────────────────
# Fusion classifier head
# ─────────────────────────────────────────────────────────────────────────

class FusionClassifier(nn.Module):
    """
    Takes concatenated audio + video embeddings and predicts stutter class.

    Parameters
    ----------
    audio_dim    : dimensionality of the audio embedding
    video_dim    : dimensionality of the video embedding
    fusion_dim   : internal dimension of the fusion MLP
    num_classes  : number of output classes
    dropout      : dropout probability
    """

    def __init__(
        self,
        audio_dim:   int   = AUDIO_EMBED_DIM,
        video_dim:   int   = VIDEO_EMBED_DIM,
        fusion_dim:  int   = FUSION_DIM,
        num_classes: int   = NUM_CLASSES,
        dropout:     float = DROPOUT,
    ):
        super().__init__()

        in_dim = audio_dim + video_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(fusion_dim // 2, num_classes)

    def forward(
        self,
        audio_emb: torch.Tensor,
        video_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        audio_emb : (B, audio_dim)
        video_emb : (B, video_dim)

        Returns
        -------
        logits : (B, num_classes)
        """
        fused  = torch.cat([audio_emb, video_emb], dim=-1)   # (B, audio+video)
        hidden = self.net(fused)                              # (B, fusion_dim//2)
        return self.classifier(hidden)                        # (B, num_classes)


# ─────────────────────────────────────────────────────────────────────────
# Full end-to-end model (Wav2Vec2 + GRU fusion)
# ─────────────────────────────────────────────────────────────────────────

class StutterDetectionModel(nn.Module):
    """
    End-to-end multimodal stutter detection.

    Audio   : Wav2Vec2Branch   (raw waveform input)
    Video   : FacialGRU        (landmark sequence input)
    Fusion  : FusionClassifier

    Parameters
    ----------
    freeze_wav2vec2_layers : int
        Number of Wav2Vec2 encoder layers to freeze (default 6 of 12).
    audio_only : bool
        If True, the video branch output is zeroed out (audio-only ablation).
    video_only : bool
        If True, the audio branch output is zeroed out (video-only ablation).
    """

    def __init__(
        self,
        freeze_wav2vec2_layers: int = 6,
        audio_only: bool = False,
        video_only: bool = False,
    ):
        super().__init__()

        self.audio_only = audio_only
        self.video_only = video_only

        self.audio_branch = Wav2Vec2Branch(
            embed_dim=AUDIO_EMBED_DIM,
            freeze_layers=freeze_wav2vec2_layers,
        )
        self.video_branch = FacialGRU(embed_dim=VIDEO_EMBED_DIM)
        self.fusion       = FusionClassifier()

    def forward(
        self,
        input_values:    torch.Tensor,
        landmark_seq:    torch.Tensor,
        attention_mask:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_values  : (B, T_audio) – raw 16-kHz waveform
        landmark_seq  : (B, T_frames, NUM_FACE_FEATURES) – facial landmarks
        attention_mask: (B, T_audio) optional mask for Wav2Vec2

        Returns
        -------
        logits : (B, NUM_CLASSES)
        """
        # Audio embedding
        if self.video_only:
            audio_emb = torch.zeros(
                input_values.size(0), AUDIO_EMBED_DIM,
                device=input_values.device
            )
        else:
            audio_emb = self.audio_branch(input_values, attention_mask)

        # Video embedding
        if self.audio_only:
            video_emb = torch.zeros(
                landmark_seq.size(0), VIDEO_EMBED_DIM,
                device=landmark_seq.device
            )
        else:
            video_emb = self.video_branch(landmark_seq)

        return self.fusion(audio_emb, video_emb)

    def predict_proba(
        self,
        input_values:   torch.Tensor,
        landmark_seq:   torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns softmax probabilities (B, NUM_CLASSES)."""
        logits = self.forward(input_values, landmark_seq, attention_mask)
        return F.softmax(logits, dim=-1)


# ─────────────────────────────────────────────────────────────────────────
# CNN + MLP variant (no Wav2Vec2 – faster training)
# ─────────────────────────────────────────────────────────────────────────

class StutterDetectionCNN(nn.Module):
    """
    Faster variant using SpectrogramCNN + FacialMLP.

    Audio   : SpectrogramCNN  (log-Mel spectrogram input)
    Video   : FacialMLP       (stat-pooled landmark vector)
    Fusion  : FusionClassifier
    """

    def __init__(self):
        super().__init__()
        self.audio_branch = SpectrogramCNN(embed_dim=AUDIO_EMBED_DIM)
        self.video_branch = FacialMLP(embed_dim=VIDEO_EMBED_DIM)
        self.fusion       = FusionClassifier()

    def forward(
        self,
        spectrogram:  torch.Tensor,
        landmark_vec: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        spectrogram  : (B, 1, N_MELS, T)
        landmark_vec : (B, 80) – stat-pooled facial features

        Returns
        -------
        logits : (B, NUM_CLASSES)
        """
        audio_emb = self.audio_branch(spectrogram)
        video_emb = self.video_branch(landmark_vec)
        return self.fusion(audio_emb, video_emb)

    def predict_proba(
        self,
        spectrogram:  torch.Tensor,
        landmark_vec: torch.Tensor,
    ) -> torch.Tensor:
        return F.softmax(self.forward(spectrogram, landmark_vec), dim=-1)
