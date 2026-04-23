"""Evaluate the trained Restormer on BMVC_image_quality_test_data (PSNR).

Mirrors `classical/eval_iq.py` but swaps Wiener for Restormer. Results go to a
separate output dir so `compare.py` can overlay both curves.

Usage (from repo root):
    python3 restormer/eval_iq.py
    python3 restormer/eval_iq.py --ckpt restormer/outputs/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "classical"))

from wienerfiltering import open_pic  # reuse the same loader as the Wiener side
from metrics import psnr
from inference import load_checkpoint, pick_device, restore_np


def list_noise_levels(root: str):
    out = [
        name
        for name in os.listdir(root)
        if re.match(r"^n_\d+$", name) and os.path.isdir(os.path.join(root, name))
    ]
    out.sort()
    return out


def id_map(folder: str) -> dict:
    d = {}
    for name in os.listdir(folder):
        if not name.lower().endswith(".png"):
            continue
        m = re.match(r"(\d+)", name)
        if m is None:
            continue
        d[m.group(1)] = name
    return d


def evaluate_level(model, device, root, noise_folder, orig_dir, limit):
    blur_dir = os.path.join(root, noise_folder)
    blur_m = id_map(blur_dir)
    orig_m = id_map(orig_dir)

    common = sorted(set(blur_m) & set(orig_m))
    if limit is not None:
        common = common[:limit]

    psnr_blur = []
    psnr_rest = []

    for img_id in common:
        try:
            blur = open_pic(os.path.join(blur_dir, blur_m[img_id]))
            orig = open_pic(os.path.join(orig_dir, orig_m[img_id]))
        except Exception as e:
            print(f"  WARN: {noise_folder} id={img_id} load failed: {e}")
            continue

        if blur.shape != orig.shape:
            print(f"  WARN: {noise_folder} id={img_id} shape mismatch; skipping")
            continue

        try:
            restored = restore_np(model, blur, device)
        except Exception as e:
            print(f"  WARN: {noise_folder} id={img_id} restore failed: {e}")
            continue

        psnr_blur.append(psnr(orig, blur))
        psnr_rest.append(psnr(orig, restored))

    if not psnr_blur:
        return None

    return {
        "n": len(psnr_blur),
        "psnr_blur": float(np.mean(psnr_blur)),
        "psnr_restormer": float(np.mean(psnr_rest)),
    }


def run(args):
    device = pick_device(args.device)
    print(f"Loading Restormer from {args.ckpt} on {device}")
    model = load_checkpoint(args.ckpt, device, variant_override=args.variant)

    os.makedirs(args.output_dir, exist_ok=True)
    levels = list_noise_levels(args.root)
    if args.noise_levels:
        filt = set(n.strip() for n in args.noise_levels.split(","))
        levels = [l for l in levels if l in filt]
    if not levels:
        print(f"No noise-level folders in {args.root}"); sys.exit(1)

    rows = []
    for lvl in levels:
        print(f"\n=== {lvl} ===")
        res = evaluate_level(model, device, args.root, lvl, args.orig_dir, args.limit)
        if res is None:
            print(f"  No images evaluated for {lvl}"); continue
        gain = res["psnr_restormer"] - res["psnr_blur"]
        print(f"  n={res['n']}  PSNR blur={res['psnr_blur']:.2f} dB "
              f"PSNR restormer={res['psnr_restormer']:.2f} dB gain={gain:+.2f} dB")
        rows.append({
            "noise_level": lvl,
            "n": res["n"],
            "psnr_blur": f"{res['psnr_blur']:.4f}",
            "psnr_restormer": f"{res['psnr_restormer']:.4f}",
            "psnr_gain": f"{gain:.4f}",
        })

    csv_path = os.path.join(args.output_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "noise_level", "n", "psnr_blur", "psnr_restormer", "psnr_gain",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {csv_path}")

    if rows:
        xs = [int(r["noise_level"].split("_")[1]) for r in rows]
        plt.figure(figsize=(8, 5))
        plt.plot(xs, [float(r["psnr_blur"]) for r in rows],
                 marker="o", label="Blurred input (baseline)")
        plt.plot(xs, [float(r["psnr_restormer"]) for r in rows],
                 marker="^", label="Restormer restored")
        plt.xlabel("Noise level (n_XX index)")
        plt.ylabel("PSNR (dB)")
        plt.title("Image-quality test: Restormer PSNR vs noise level")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        p = os.path.join(args.output_dir, "psnr_vs_noise.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot saved to {p}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="data/BMVC_image_quality_test_data")
    p.add_argument("--orig-dir", default=None)
    p.add_argument("--ckpt", default="restormer/outputs/checkpoints/best.pt")
    p.add_argument("--variant", default=None,
                   help="Override variant stored in ckpt (rarely needed)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--noise-levels", default=None)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--output-dir", default="restormer/outputs/eval_iq")
    a = p.parse_args()
    if a.orig_dir is None:
        a.orig_dir = os.path.join(a.root, "orig")
    return a


if __name__ == "__main__":
    run(parse_args())
