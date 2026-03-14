"""
training/train.py
──────────────────
Entry-point script to train the Stutter Detection Model.

Usage
─────
  # Audio-only training (SEP-28k; no video data needed)
  python training/train.py --mode audio_only

  # Multimodal training (requires pre-extracted .npy landmark files)
  python training/train.py --mode full

  # Use the faster CNN variant instead of Wav2Vec2
  python training/train.py --model cnn --mode audio_only

Options
───────
  --mode       audio_only | full          (default: audio_only)
  --model      wav2vec2   | cnn           (default: wav2vec2)
  --epochs     N                          (default: config.NUM_EPOCHS)
  --batch      N                          (default: config.BATCH_SIZE)
  --freeze     N  Wav2Vec2 layers frozen  (default: 6)
  --csv        path to processed CSV      (default: auto)
  --checkpoint output model name          (default: stutter_model)
"""

import sys
import argparse
import os
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import NUM_EPOCHS, BATCH_SIZE, MODELS_DIR
from models.fusion_model import StutterDetectionModel, StutterDetectionCNN
from training.dataset    import build_dataloaders
from training.trainer    import Trainer


# ─────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train Stutter Detection Model")
    parser.add_argument("--mode",       default="audio_only",
                        choices=["audio_only", "full"])
    parser.add_argument("--model",      default="wav2vec2",
                        choices=["wav2vec2", "cnn"])
    parser.add_argument("--epochs",     type=int,   default=NUM_EPOCHS)
    parser.add_argument("--batch",      type=int,   default=BATCH_SIZE)
    parser.add_argument("--freeze",     type=int,   default=6)
    parser.add_argument("--workers",    type=int,
                        default=(0 if os.name == "nt" else 2))
    parser.add_argument("--csv",        type=str,   default=None)
    parser.add_argument("--checkpoint", type=str,   default="stutter_model")
    parser.add_argument("--patience",   type=int,   default=None,
                        help="early-stopping patience (overrides config.PATIENCE)")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="disable early stopping and always run all epochs")
    parser.add_argument("--resume", action="store_true",
                        help="load existing checkpoint before training")
    return parser.parse_args()


def main():
    args   = parse_args()

    print("\n" + "="*60)
    print("  Stutter Detection – Training")
    print(f"  model={args.model}  mode={args.mode}  "
          f"epochs={args.epochs}  batch={args.batch}")
    print("="*60 + "\n")

    # ── Build DataLoaders ─────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(
        csv_path   = args.csv,
        batch_size = args.batch,
        mode       = args.mode,
        num_workers= args.workers,
    )

    # ── Select model ─────────────────────────────────────────────────────
    if args.model == "wav2vec2":
        model = StutterDetectionModel(
            freeze_wav2vec2_layers=args.freeze,
            audio_only=(args.mode == "audio_only"),
        )
    else:
        model = StutterDetectionCNN()

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters → total: {total_params:,}   trainable: {trainable_params:,}\n")

    # ── Train ─────────────────────────────────────────────────────────────
    # determine patience / early-stop behaviour
    patience = args.patience if args.patience is not None else None
    early_stop = not args.no_early_stop

    trainer = Trainer(
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        mode            = args.mode,
        checkpoint_name = args.checkpoint,
        num_epochs      = args.epochs,
        patience        = patience,
        early_stop      = early_stop,
    )

    # option to resume from existing checkpoint
    if args.resume:
        ckpt = MODELS_DIR / f"{args.checkpoint}.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            print(f"Resuming training from checkpoint: {ckpt}")
        else:
            print(f"No checkpoint found at {ckpt}; starting from scratch")

    history = trainer.train()

    # ── Final test-set evaluation ─────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation …")
    best_path = MODELS_DIR / f"{args.checkpoint}.pt"
    model.load_state_dict(torch.load(best_path, map_location="cpu"))
    trainer.evaluate(test_loader)


if __name__ == "__main__":
    main()
