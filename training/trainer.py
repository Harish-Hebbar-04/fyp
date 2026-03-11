"""
training/trainer.py
────────────────────
Improved training loop with:
  • Focal Loss (γ=2, squared-inverse-freq weights) – aggressive class balance
  • MixUp augmentation   – interpolates pairs, especially benefits minorities
  • SpecAugment on mel   – freq/time masking per batch during training
  • AdamW + CosineAnnealingWarmRestarts (T_0=20)
  • Linear LR warm-up (first 5 epochs)
  • Gradient clipping (max_norm=1.0)
  • Early stopping on best val macro-F1
  • Per-epoch metrics printed (accuracy, macro-F1, lr)
"""

import sys
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import f1_score, accuracy_score, classification_report


# ─────────────────────────────────────────────────────────────────────────
# Focal Loss  (handles severe class imbalance far better than weighted CE)
# ─────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)

    Parameters
    ----------
    weight : (C,) class weights (inverse-frequency)  → α_t
    gamma  : focusing parameter; 2.0 is a strong default
    """
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight   # kept on whatever device it was created on

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(logits.device) if self.weight is not None else None
        ce     = F.cross_entropy(logits, targets, weight=weight, reduction="none")
        pt     = torch.exp(-ce)                       # probability of correct class
        loss   = (1.0 - pt) ** self.gamma * ce
        return loss.mean()

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY, PATIENCE,
    MODELS_DIR, DEVICE, NUM_CLASSES, ID2LABEL,
)
from training.augmentation import spec_augment
from models.fusion_model import StutterDetectionCNN


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_class_weights(train_loader: DataLoader,
                           num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Compute class weights from frequency in training set.

    We use the same exponent as the sampler so that loss weighting and
    sampling behaviour are consistent.  Set :data:`config.CLASS_WEIGHT_EXP`
    to 1.0 for simple inverse‑frequency, or higher for stronger minority
    emphasis.
    """
    from config import CLASS_WEIGHT_EXP
    counts = torch.zeros(num_classes)
    for batch in train_loader:
        labels = batch["label"]
        for c in range(num_classes):
            counts[c] += (labels == c).sum()
    weights = (1.0 / (counts + 1e-6)) ** CLASS_WEIGHT_EXP
    return weights / weights.sum() * num_classes   # normalised


def mixup_batch(
    mel: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    MixUp – blends two random mel-spectrogram samples and their labels.
    Returns (mixed_mel, labels_a, labels_b, lambda).
    """
    lam = float(np.random.beta(alpha, alpha))
    B   = mel.size(0)
    idx = torch.randperm(B, device=mel.device)
    return lam * mel + (1.0 - lam) * mel[idx], labels, labels[idx], lam


def apply_spec_augment_batch(mel: torch.Tensor, training: bool) -> torch.Tensor:
    """
    Apply SpecAugment (freq+time masking) to a (B, 1, N_MELS, T) batch.
    No-op at eval time.
    """
    if not training:
        return mel
    out = mel.clone()
    for i in range(mel.size(0)):
        if random.random() < 0.7:   # skip ~30% of samples
            arr = out[i, 0].cpu().numpy()   # (N_MELS, T)
            arr = spec_augment(
                arr,
                freq_mask_param=20,
                time_mask_param=40,
                num_freq_masks=2,
                num_time_masks=2,
            )
            out[i, 0] = torch.from_numpy(arr).to(mel.device)
    return out


# ─────────────────────────────────────────────────────────────────────────
# Main Trainer
# ─────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Encapsulates the full training loop.

    Parameters
    ----------
    model        : nn.Module  – StutterDetectionModel or StutterDetectionCNN
    train_loader : DataLoader
    val_loader   : DataLoader
    mode         : 'audio_only' | 'full'
    checkpoint_name : filename (no extension) for saved model weights
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        mode: str                = "audio_only",
        checkpoint_name: str     = "stutter_model",
        lr: float                = LEARNING_RATE,
        num_epochs: int          = NUM_EPOCHS,
        patience: int | None     = None,            # allow CLI override
        warmup_epochs: int       = 5,
        mixup_alpha: float       = 0.4,
        early_stop: bool         = True,            # optionally disable
    ):
        self.device        = get_device()
        self.model         = model.to(self.device)
        self.train_dl      = train_loader
        self.val_dl        = val_loader
        self.mode          = mode
        self.num_epochs    = num_epochs
        # if caller passes None, fall back to config.PATIENCE
        self.patience      = patience if patience is not None else PATIENCE
        self.early_stop    = early_stop
        self.warmup_epochs = warmup_epochs
        self.mixup_alpha   = mixup_alpha
        self.save_path     = MODELS_DIR / f"{checkpoint_name}.pt"
        self.peak_lr       = lr

        # ── Loss ─────────────────────────────────────────────────────────
        print("Computing class weights …", end=" ")
        weights = compute_class_weights(train_loader).to(self.device)
        print(weights.cpu().numpy().round(4).tolist())
        self.criterion = FocalLoss(weight=weights, gamma=2.0)
        print("Loss : Focal (γ=2.0) + squared-inverse-freq class weights")

        # ── Optimiser ─────────────────────────────────────────────────────
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=WEIGHT_DECAY,
        )
        # Cosine warm-restarts – new cycle every 20 epochs
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=1, eta_min=lr * 0.01
        )

        self.history = {
            "train_loss": [], "val_loss": [],
            "val_acc":    [], "val_f1":   [],
        }

        # CNN detection
        self._is_cnn = isinstance(model, StutterDetectionCNN)

    def _warmup_lr(self, epoch: int):
        """Linear warm-up: scale LR from 0 → peak over warmup_epochs."""
        scale = epoch / max(self.warmup_epochs, 1)
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.peak_lr * scale

    # ─────────────────────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict:
        self.model.train() if train else self.model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                labels = batch["label"].to(self.device)

                if self._is_cnn:
                    # mel_spec is pre-cached in StutterDataset ↓ no librosa per batch
                    mel   = batch["mel_spec"].to(self.device)  # (B, 1, N_MELS, T)
                    mel   = apply_spec_augment_batch(mel, training=train)
                    dummy = torch.zeros(labels.size(0), 4 * 20 * 2,
                                        device=self.device)

                    if train and self.mixup_alpha > 0:
                        mel, la, lb, lam = mixup_batch(mel, labels,
                                                        self.mixup_alpha)
                        logits = self.model(mel, dummy)
                        loss   = (lam       * self.criterion(logits, la)
                                  + (1-lam) * self.criterion(logits, lb))
                    else:
                        logits = self.model(mel, dummy)
                        loss   = self.criterion(logits, labels)

                elif self.mode == "audio_only":
                    logits = self.model(
                        input_values=batch["input_values"].to(self.device),
                        landmark_seq=torch.zeros(
                            labels.size(0), 1, 1, device=self.device),
                    )
                    loss = self.criterion(logits, labels)
                else:
                    logits = self.model(
                        input_values=batch["input_values"].to(self.device),
                        landmark_seq=batch["landmark_seq"].to(self.device),
                    )
                    loss = self.criterion(logits, labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        acc      = accuracy_score(all_labels, all_preds)
        f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return {"loss": avg_loss, "acc": acc, "f1": f1,
                "preds": all_preds, "labels": all_labels}

    # ─────────────────────────────────────────────────────────────────────

    def train(self):
        print(f"\n{'='*60}")
        print(f"  Training on {self.device}  |  epochs={self.num_epochs}"
              f"  patience={self.patience}")
        print(f"  MixUp α={self.mixup_alpha}  warmup={self.warmup_epochs} ep")
        print(f"{'='*60}")

        best_val_f1 = -1.0
        no_improve  = 0

        for epoch in range(1, self.num_epochs + 1):
            # Linear LR warm-up overrides scheduler for first N epochs
            if epoch <= self.warmup_epochs:
                self._warmup_lr(epoch)

            t0 = time.time()
            train_metrics = self._run_epoch(self.train_dl, train=True)
            val_metrics   = self._run_epoch(self.val_dl,   train=False)

            # Cosine warm-restart scheduler kicks in after warm-up
            if epoch > self.warmup_epochs:
                self.scheduler.step(epoch - self.warmup_epochs)

            elapsed = time.time() - t0
            cur_lr  = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["acc"])
            self.history["val_f1"].append(val_metrics["f1"])

            print(
                f"Epoch {epoch:03d}/{self.num_epochs}  [{elapsed:.1f}s]  "
                f"train_loss={train_metrics['loss']:.4f}  "
                f"val_loss={val_metrics['loss']:.4f}  "
                f"val_acc={val_metrics['acc']:.4f}  "
                f"val_f1={val_metrics['f1']:.4f}  "
                f"lr={cur_lr:.2e}"
            )

            # ── Checkpoint on best val macro-F1 ──────────────────────
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                no_improve  = 0
                torch.save(self.model.state_dict(), self.save_path)
                print(f"          ✓ Best saved  val_f1={best_val_f1:.4f}"
                      f" → {self.save_path}")
            else:
                no_improve += 1
                if self.early_stop and no_improve >= self.patience:
                    print(f"\n  Early stopping at epoch {epoch} "
                          f"(no F1 improvement for {self.patience} epochs).")
                    break

        print(f"\nTraining complete.  Best val macro-F1 = {best_val_f1:.4f}")
        return self.history

    # ─────────────────────────────────────────────────────────────────────

    def evaluate(self, loader: DataLoader) -> dict:
        """Full evaluation including per-class report."""
        metrics = self._run_epoch(loader, train=False)
        labels  = [ID2LABEL[i] for i in metrics["labels"]]
        preds   = [ID2LABEL[i] for i in metrics["preds"]]
        print("\nClassification Report:")
        print(classification_report(labels, preds, zero_division=0))
        return metrics
