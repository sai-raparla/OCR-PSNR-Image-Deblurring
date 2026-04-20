"""Evaluate Wiener deblurring on BMVC_image_quality_test_data across all noise
levels using PSNR as the image-quality metric.

Usage (from repo root):
    python3 classical/eval_iq.py
    python3 classical/eval_iq.py --k 0.01
    python3 classical/eval_iq.py --k-summary classical/outputs/tuning/tune_k_summary.json

Writes:
    classical/outputs/eval_iq/results.csv
    classical/outputs/eval_iq/psnr_vs_noise.png
"""

import argparse
import csv
import json
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from wienerfiltering import open_pic
from metrics import psnr, wiener_restore_arr


def list_noise_levels(root: str):
    """Return sorted list of n_XX subfolder names."""
    out = []
    for name in os.listdir(root):
        if re.match(r"^n_\d+$", name) and os.path.isdir(os.path.join(root, name)):
            out.append(name)
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


def resolve_k(args) -> float:
    if args.k is not None:
        return float(args.k)
    if args.k_summary is not None:
        with open(args.k_summary) as f:
            return float(json.load(f)["best_k"])
    default_summary = "classical/outputs/tuning/tune_k_summary.json"
    if os.path.exists(default_summary):
        with open(default_summary) as f:
            best = float(json.load(f)["best_k"])
            print(f"Using best_k={best} from {default_summary}")
            return best
    print("No --k or --k-summary provided and no summary found; defaulting to k=0.01")
    return 0.01


def evaluate_level(root: str, noise_folder: str, psf_dir: str, orig_dir: str,
                   k: float, limit=None):
    blur_dir = os.path.join(root, noise_folder)
    blur_m = id_map(blur_dir)
    psf_m = id_map(psf_dir)
    orig_m = id_map(orig_dir)

    common = sorted(set(blur_m) & set(psf_m) & set(orig_m))
    if limit is not None:
        common = common[:limit]

    psnr_blur = []
    psnr_wiener = []

    for img_id in common:
        try:
            blur = open_pic(os.path.join(blur_dir, blur_m[img_id]))
            psf = open_pic(os.path.join(psf_dir, psf_m[img_id]))
            orig = open_pic(os.path.join(orig_dir, orig_m[img_id]))
        except Exception as e:
            print(f"  WARN: {noise_folder} id={img_id} load failed: {e}")
            continue

        if blur.shape != orig.shape:
            print(f"  WARN: {noise_folder} id={img_id} shape mismatch,"
                  f" blur={blur.shape} orig={orig.shape}; skipping")
            continue

        try:
            restored = wiener_restore_arr(blur, psf, k)
        except Exception as e:
            print(f"  WARN: {noise_folder} id={img_id} wiener failed: {e}")
            continue

        psnr_blur.append(psnr(orig, blur))
        psnr_wiener.append(psnr(orig, restored))

    if not psnr_blur:
        return None

    return {
        "n": len(psnr_blur),
        "psnr_blur": float(np.mean(psnr_blur)),
        "psnr_wiener": float(np.mean(psnr_wiener)),
    }


def run(root, psf_dir, orig_dir, k, output_dir, limit, noise_levels_filter):
    os.makedirs(output_dir, exist_ok=True)

    levels = list_noise_levels(root)
    if noise_levels_filter:
        levels = [n for n in levels if n in noise_levels_filter]
    if not levels:
        print(f"No noise-level folders found under {root}")
        sys.exit(1)

    print(f"Evaluating k={k} over noise levels: {levels}")
    print("-" * 70)

    rows = []
    for lvl in levels:
        print(f"\n=== {lvl} ===")
        res = evaluate_level(root, lvl, psf_dir, orig_dir, k, limit=limit)
        if res is None:
            print(f"  No images evaluated for {lvl}")
            continue
        print(f"  n={res['n']}  PSNR blur={res['psnr_blur']:.2f} dB"
              f"  PSNR wiener={res['psnr_wiener']:.2f} dB"
              f"  gain={res['psnr_wiener'] - res['psnr_blur']:+.2f} dB")
        rows.append({
            "noise_level": lvl,
            "n": res["n"],
            "psnr_blur": f"{res['psnr_blur']:.4f}",
            "psnr_wiener": f"{res['psnr_wiener']:.4f}",
            "psnr_gain": f"{res['psnr_wiener'] - res['psnr_blur']:.4f}",
        })

    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["noise_level", "n", "psnr_blur", "psnr_wiener", "psnr_gain"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {csv_path}")

    if rows:
        xs = [int(r["noise_level"].split("_")[1]) for r in rows]
        y_blur = [float(r["psnr_blur"]) for r in rows]
        y_wien = [float(r["psnr_wiener"]) for r in rows]

        plt.figure(figsize=(8, 5))
        plt.plot(xs, y_blur, marker="o", label="Blurred input (baseline)")
        plt.plot(xs, y_wien, marker="s", label=f"Wiener restored (k={k})")
        plt.xlabel("Noise level (n_XX index)")
        plt.ylabel("PSNR (dB)")
        plt.title("Image-quality test: PSNR vs noise level")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(output_dir, "psnr_vs_noise.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Plot saved to {plot_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="data/BMVC_image_quality_test_data")
    p.add_argument("--psf-dir", default=None,
                   help="Defaults to <root>/psf")
    p.add_argument("--orig-dir", default=None,
                   help="Defaults to <root>/orig")
    p.add_argument("--k", type=float, default=None)
    p.add_argument("--k-summary", default=None,
                   help="Path to tune_k_summary.json (overrides --k if --k not set)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap images per noise level (for quick runs)")
    p.add_argument("--noise-levels", default=None,
                   help="Comma-separated list of n_XX to restrict (e.g. 'n_00,n_01')")
    p.add_argument("--output-dir", default="classical/outputs/eval_iq")
    a = p.parse_args()

    psf_dir = a.psf_dir or os.path.join(a.root, "psf")
    orig_dir = a.orig_dir or os.path.join(a.root, "orig")
    filt = set(x.strip() for x in a.noise_levels.split(",")) if a.noise_levels else None

    k = resolve_k(a)
    run(a.root, psf_dir, orig_dir, k, a.output_dir, a.limit, filt)


if __name__ == "__main__":
    main()
