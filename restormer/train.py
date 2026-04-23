"""Train Restormer on BMVC_image_data.

Usage (from repo root):
    # Small variant, sensible defaults for MPS / CPU:
    python3 restormer/train.py --variant small --epochs 30 --batch-size 8

    # Paper-sized variant, CUDA recommended:
    python3 restormer/train.py --variant base --epochs 200 --batch-size 16 \
                               --patch-size 128 --lr 3e-4

Checkpoints are written to `restormer/outputs/checkpoints/`.
The best-val (lowest val L1) checkpoint is saved as `best.pt` and is the one
the eval scripts load by default.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dataset import BMVCPairedDataset
from model import build_model, count_params


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def pick_device(pref: str) -> torch.device:
    pref = pref.lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


def psnr_batch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> float:
    pred = pred.clamp(0.0, 1.0)
    mse = F.mse_loss(pred, target).item()
    if mse < eps:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup: int, min_lr: float):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args):
    device = pick_device(args.device)
    print(f"Using device: {device}")

    dataset = BMVCPairedDataset(
        blur_dir=args.blur_dir,
        orig_dir=args.orig_dir,
        patch_size=args.patch_size,
        augment=True,
        limit=args.limit,
        seed=args.seed,
    )
    print(f"Dataset: {len(dataset)} paired images")

    val_size = max(1, int(len(dataset) * args.val_frac))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"  train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(args.variant).to(device)
    n_params = count_params(model)
    print(f"Restormer-{args.variant}: {n_params/1e6:.2f} M trainable params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_amp = (device.type == "cuda") and args.amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(args.warmup_steps, max(1, total_steps // 10))
    print(f"Total optim steps: {total_steps}  (warmup {warmup_steps})")

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    start_epoch = 0
    best_val = math.inf
    global_step = 0

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optim"])
        start_epoch = state.get("epoch", 0)
        best_val = state.get("best_val", math.inf)
        global_step = state.get("global_step", 0)

    log_path = os.path.join(args.output_dir, "train_log.csv")
    log_exists = os.path.exists(log_path)
    log_file = open(log_path, "a")
    if not log_exists:
        log_file.write("epoch,train_loss,val_loss,val_psnr,lr,elapsed_s\n")
        log_file.flush()

    criterion = nn.L1Loss()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t_epoch = time.time()
        running = 0.0
        n = 0

        for blur, orig in train_loader:
            blur = blur.to(device, non_blocking=True)
            orig = orig.to(device, non_blocking=True)

            lr = cosine_lr(global_step, total_steps, args.lr, warmup_steps, args.min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(blur)
                loss = criterion(pred, orig)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            running += loss.item() * blur.size(0)
            n += blur.size(0)
            global_step += 1

            if global_step % args.log_every == 0:
                print(f"  epoch {epoch+1} step {global_step}/{total_steps} "
                      f"loss={loss.item():.4f} lr={lr:.2e}")

        train_loss = running / max(1, n)

        val_loss, val_psnr = evaluate(model, val_loader, device, criterion)

        elapsed = time.time() - t_epoch
        print(f"[epoch {epoch+1}/{args.epochs}] "
              f"train L1={train_loss:.4f}  val L1={val_loss:.4f}  val PSNR={val_psnr:.2f} dB  "
              f"({elapsed:.1f}s)")
        log_file.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{val_psnr:.4f},{lr:.6e},{elapsed:.1f}\n")
        log_file.flush()

        ckpt = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_val": best_val,
            "global_step": global_step,
            "variant": args.variant,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(ckpt_dir, "last.pt"))

        if val_loss < best_val:
            best_val = val_loss
            ckpt["best_val"] = best_val
            torch.save(ckpt, os.path.join(ckpt_dir, "best.pt"))
            print(f"  -> new best (val L1={best_val:.4f}), saved best.pt")

    log_file.close()

    summary = {
        "variant": args.variant,
        "epochs": args.epochs,
        "best_val_l1": best_val,
        "params_millions": n_params / 1e6,
        "train_size": train_size,
        "val_size": val_size,
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
    }
    with open(os.path.join(args.output_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTraining done. Summary: {summary}")


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> tuple:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    n = 0
    for blur, orig in loader:
        blur = blur.to(device)
        orig = orig.to(device)
        pred = model(blur)
        total_loss += criterion(pred, orig).item() * blur.size(0)
        total_psnr += psnr_batch(pred, orig) * blur.size(0)
        n += blur.size(0)
    return total_loss / max(1, n), total_psnr / max(1, n)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Train Restormer on BMVC_image_data")
    p.add_argument("--blur-dir", default="data/BMVC_image_data/blur")
    p.add_argument("--orig-dir", default="data/BMVC_image_data/orig")
    p.add_argument("--output-dir", default="restormer/outputs")
    p.add_argument("--variant", choices=["small", "base"], default="small")

    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of training pairs (fast smoke-tests)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--amp", action="store_true", help="AMP fp16 (CUDA only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


if __name__ == "__main__":
    torch.manual_seed(0)
    train(parse_args())
