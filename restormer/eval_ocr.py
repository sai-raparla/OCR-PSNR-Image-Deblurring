"""Evaluate the trained Restormer on BMVC_OCR_test_data (CER/WER).

Mirrors `classical/eval_ocr.py` but swaps Wiener for Restormer, so the output
CSVs and plots are directly comparable.

Usage (from repo root):
    python3 restormer/eval_ocr.py
    python3 restormer/eval_ocr.py --ckpt restormer/outputs/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import numpy as np
import pytesseract
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "classical"))
sys.path.insert(0, os.path.join(REPO, "ocr"))

from wienerfiltering import open_pic
from tesseract_eval import character_error_rate, word_error_rate, run_ocr
from inference import load_checkpoint, pick_device, restore_np


def arr_to_pil(arr: np.ndarray) -> Image.Image:
    a = np.clip(arr, 0.0, 1.0) * 255.0
    return Image.fromarray(a.astype(np.uint8))


def ocr_pil(img: Image.Image, psm: int) -> str:
    return pytesseract.image_to_string(img, config=f"--psm {psm}").strip()


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


def ocr_all_origs(orig_dir: str, ids, psm: int):
    print(f"OCR-ing {len(ids)} orig references...")
    t0 = time.time()
    orig_map = id_map(orig_dir)
    texts = {}
    for i, img_id in enumerate(ids, 1):
        try:
            p = os.path.join(orig_dir, orig_map[img_id])
            texts[img_id] = run_ocr(p, psm)["text"]
        except Exception as e:
            texts[img_id] = ""
            print(f"  WARN: orig {img_id}: {e}")
        if i % 20 == 0 or i == len(ids):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (len(ids) - i) / rate if rate > 0 else 0.0
            print(f"  [{i}/{len(ids)}] {rate:.2f} img/s | ETA: {eta:.0f}s")
    return texts


def evaluate_level(model, device, root, noise_folder, ids, orig_texts, psm, limit):
    blur_dir = os.path.join(root, noise_folder)
    blur_m = id_map(blur_dir)

    common = [i for i in ids if i in blur_m]
    if limit is not None:
        common = common[:limit]

    rows = []
    cer_blur_list, cer_rest_list = [], []
    wer_blur_list, wer_rest_list = [], []
    t0 = time.time()
    n = len(common)

    for idx, img_id in enumerate(common, 1):
        try:
            blur_arr = open_pic(os.path.join(blur_dir, blur_m[img_id]))
        except Exception as e:
            print(f"  WARN: {noise_folder} id={img_id} load failed: {e}")
            continue

        try:
            blur_text = ocr_pil(arr_to_pil(blur_arr), psm)
            restored = restore_np(model, blur_arr, device)
            rest_text = ocr_pil(arr_to_pil(restored), psm)
        except Exception as e:
            print(f"  WARN: {noise_folder} id={img_id} OCR/restore failed: {e}")
            continue

        ref = orig_texts.get(img_id, "")
        cer_b = character_error_rate(ref, blur_text)
        cer_r = character_error_rate(ref, rest_text)
        wer_b = word_error_rate(ref, blur_text)
        wer_r = word_error_rate(ref, rest_text)

        cer_blur_list.append(cer_b)
        cer_rest_list.append(cer_r)
        wer_blur_list.append(wer_b)
        wer_rest_list.append(wer_r)

        rows.append({
            "noise_level": noise_folder,
            "image_id": img_id,
            "cer_blur": f"{cer_b:.4f}",
            "cer_restormer": f"{cer_r:.4f}",
            "wer_blur": f"{wer_b:.4f}",
            "wer_restormer": f"{wer_r:.4f}",
            "len_orig": len(ref),
            "len_blur": len(blur_text),
            "len_restormer": len(rest_text),
        })

        if idx % 20 == 0 or idx == n:
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else 0.0
            eta = (n - idx) / rate if rate > 0 else 0.0
            print(f"  [{idx}/{n}] {rate:.2f} img/s | ETA: {eta:.0f}s")

    if not rows:
        return None, []

    return {
        "n": len(rows),
        "cer_blur": float(np.mean(cer_blur_list)),
        "cer_restormer": float(np.mean(cer_rest_list)),
        "wer_blur": float(np.mean(wer_blur_list)),
        "wer_restormer": float(np.mean(wer_rest_list)),
    }, rows


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

    orig_ids = sorted(id_map(args.orig_dir).keys())
    if args.limit is not None:
        orig_ids = orig_ids[:args.limit]

    print(f"Evaluating Restormer over noise levels: {levels}")
    print(f"Tesseract PSM: {args.psm}")
    print("-" * 70)

    orig_texts = ocr_all_origs(args.orig_dir, orig_ids, args.psm)

    summary_rows = []
    per_image_rows = []

    for lvl in levels:
        print(f"\n=== {lvl} ===")
        summary, rows = evaluate_level(
            model, device, args.root, lvl, orig_ids, orig_texts, args.psm, args.limit,
        )
        if summary is None:
            print(f"  No images evaluated for {lvl}")
            continue
        print(f"  n={summary['n']}"
              f"  CER blur={summary['cer_blur']:.4f}"
              f"  CER restormer={summary['cer_restormer']:.4f}"
              f"  (gain {summary['cer_blur'] - summary['cer_restormer']:+.4f})")
        print(f"             WER blur={summary['wer_blur']:.4f}"
              f"  WER restormer={summary['wer_restormer']:.4f}")

        summary_rows.append({
            "noise_level": lvl,
            "n": summary["n"],
            "cer_blur": f"{summary['cer_blur']:.4f}",
            "cer_restormer": f"{summary['cer_restormer']:.4f}",
            "cer_gain": f"{summary['cer_blur'] - summary['cer_restormer']:.4f}",
            "wer_blur": f"{summary['wer_blur']:.4f}",
            "wer_restormer": f"{summary['wer_restormer']:.4f}",
            "wer_gain": f"{summary['wer_blur'] - summary['wer_restormer']:.4f}",
        })
        per_image_rows.extend(rows)

    csv_summary = os.path.join(args.output_dir, "results.csv")
    with open(csv_summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "noise_level", "n",
            "cer_blur", "cer_restormer", "cer_gain",
            "wer_blur", "wer_restormer", "wer_gain",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary saved to {csv_summary}")

    csv_per = os.path.join(args.output_dir, "per_image.csv")
    if per_image_rows:
        with open(csv_per, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_image_rows)
        print(f"Per-image rows saved to {csv_per}")

    if summary_rows:
        xs = [int(r["noise_level"].split("_")[1]) for r in summary_rows]

        plt.figure(figsize=(8, 5))
        plt.plot(xs, [float(r["cer_blur"]) for r in summary_rows],
                 marker="o", label="Blurred input (baseline)")
        plt.plot(xs, [float(r["cer_restormer"]) for r in summary_rows],
                 marker="^", label="Restormer restored")
        plt.xlabel("Noise level (n_XX index)")
        plt.ylabel("Character Error Rate")
        plt.title("OCR test: Restormer CER vs noise level (lower is better)")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        p1 = os.path.join(args.output_dir, "cer_vs_noise.png")
        plt.savefig(p1, dpi=150); plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(xs, [float(r["wer_blur"]) for r in summary_rows],
                 marker="o", label="Blurred input (baseline)")
        plt.plot(xs, [float(r["wer_restormer"]) for r in summary_rows],
                 marker="^", label="Restormer restored")
        plt.xlabel("Noise level (n_XX index)")
        plt.ylabel("Word Error Rate")
        plt.title("OCR test: Restormer WER vs noise level (lower is better)")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        p2 = os.path.join(args.output_dir, "wer_vs_noise.png")
        plt.savefig(p2, dpi=150); plt.close()
        print(f"Plots saved to {p1} and {p2}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="data/BMVC_OCR_test_data")
    p.add_argument("--orig-dir", default=None)
    p.add_argument("--ckpt", default="restormer/outputs/checkpoints/best.pt")
    p.add_argument("--variant", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--noise-levels", default=None)
    p.add_argument("--psm", type=int, default=6)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--output-dir", default="restormer/outputs/eval_ocr")
    a = p.parse_args()
    if a.orig_dir is None:
        a.orig_dir = os.path.join(a.root, "orig")
    return a


if __name__ == "__main__":
    run(parse_args())
