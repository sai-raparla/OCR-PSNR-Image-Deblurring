"""Evaluate Wiener deblurring on BMVC_OCR_test_data across all noise levels
using CER and WER (Tesseract OCR).

Usage (from repo root):
    python3 classical/eval_ocr.py
    python3 classical/eval_ocr.py --k 0.01
    python3 classical/eval_ocr.py --k-summary classical/outputs/tuning/tune_k_summary.json

Writes:
    classical/outputs/eval_ocr/results.csv
    classical/outputs/eval_ocr/per_image.csv
    classical/outputs/eval_ocr/cer_vs_noise.png
    classical/outputs/eval_ocr/wer_vs_noise.png
"""

import argparse
import csv
import json
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
sys.path.insert(0, os.path.join(REPO, "ocr"))

from wienerfiltering import open_pic
from metrics import wiener_restore_arr
from tesseract_eval import character_error_rate, word_error_rate, run_ocr


def arr_to_pil(arr: np.ndarray) -> Image.Image:
    a = np.clip(arr, 0.0, 1.0) * 255.0
    return Image.fromarray(a.astype(np.uint8))


def ocr_pil(img: Image.Image, psm: int) -> str:
    return pytesseract.image_to_string(img, config=f"--psm {psm}").strip()


def list_noise_levels(root: str):
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


def evaluate_level(root, noise_folder, psf_dir, ids, orig_texts, k, psm, limit):
    blur_dir = os.path.join(root, noise_folder)
    blur_m = id_map(blur_dir)
    psf_m = id_map(psf_dir)

    common = [i for i in ids if i in blur_m and i in psf_m]
    if limit is not None:
        common = common[:limit]

    rows = []
    cer_blur_list = []
    cer_wien_list = []
    wer_blur_list = []
    wer_wien_list = []
    t0 = time.time()
    n = len(common)

    for idx, img_id in enumerate(common, 1):
        try:
            blur_arr = open_pic(os.path.join(blur_dir, blur_m[img_id]))
            psf_arr = open_pic(os.path.join(psf_dir, psf_m[img_id]))
        except Exception as e:
            print(f"  WARN: {noise_folder} id={img_id} load failed: {e}")
            continue

        try:
            blur_pil = arr_to_pil(blur_arr)
            blur_text = ocr_pil(blur_pil, psm)

            restored = wiener_restore_arr(blur_arr, psf_arr, k)
            wien_pil = arr_to_pil(restored)
            wien_text = ocr_pil(wien_pil, psm)
        except Exception as e:
            print(f"  WARN: {noise_folder} id={img_id} OCR failed: {e}")
            continue

        ref = orig_texts.get(img_id, "")
        cer_blur = character_error_rate(ref, blur_text)
        cer_wien = character_error_rate(ref, wien_text)
        wer_blur = word_error_rate(ref, blur_text)
        wer_wien = word_error_rate(ref, wien_text)

        cer_blur_list.append(cer_blur)
        cer_wien_list.append(cer_wien)
        wer_blur_list.append(wer_blur)
        wer_wien_list.append(wer_wien)

        rows.append({
            "noise_level": noise_folder,
            "image_id": img_id,
            "cer_blur": f"{cer_blur:.4f}",
            "cer_wiener": f"{cer_wien:.4f}",
            "wer_blur": f"{wer_blur:.4f}",
            "wer_wiener": f"{wer_wien:.4f}",
            "len_orig": len(ref),
            "len_blur": len(blur_text),
            "len_wiener": len(wien_text),
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
        "cer_wiener": float(np.mean(cer_wien_list)),
        "wer_blur": float(np.mean(wer_blur_list)),
        "wer_wiener": float(np.mean(wer_wien_list)),
    }, rows


def run(root, psf_dir, orig_dir, k, output_dir, limit, noise_levels_filter, psm):
    os.makedirs(output_dir, exist_ok=True)

    levels = list_noise_levels(root)
    if noise_levels_filter:
        levels = [n for n in levels if n in noise_levels_filter]
    if not levels:
        print(f"No noise-level folders found under {root}")
        sys.exit(1)

    orig_ids = sorted(id_map(orig_dir).keys())
    if limit is not None:
        orig_ids = orig_ids[:limit]

    print(f"Evaluating k={k} over noise levels: {levels}")
    print(f"Tesseract PSM: {psm}")
    print("-" * 70)

    orig_texts = ocr_all_origs(orig_dir, orig_ids, psm)

    summary_rows = []
    per_image_rows = []

    for lvl in levels:
        print(f"\n=== {lvl} ===")
        summary, rows = evaluate_level(
            root, lvl, psf_dir, orig_ids, orig_texts, k, psm, limit
        )
        if summary is None:
            print(f"  No images evaluated for {lvl}")
            continue
        print(f"  n={summary['n']}"
              f"  CER blur={summary['cer_blur']:.4f}"
              f"  CER wiener={summary['cer_wiener']:.4f}"
              f"  (gain {summary['cer_blur'] - summary['cer_wiener']:+.4f})")
        print(f"             WER blur={summary['wer_blur']:.4f}"
              f"  WER wiener={summary['wer_wiener']:.4f}")

        summary_rows.append({
            "noise_level": lvl,
            "n": summary["n"],
            "cer_blur": f"{summary['cer_blur']:.4f}",
            "cer_wiener": f"{summary['cer_wiener']:.4f}",
            "cer_gain": f"{summary['cer_blur'] - summary['cer_wiener']:.4f}",
            "wer_blur": f"{summary['wer_blur']:.4f}",
            "wer_wiener": f"{summary['wer_wiener']:.4f}",
            "wer_gain": f"{summary['wer_blur'] - summary['wer_wiener']:.4f}",
        })
        per_image_rows.extend(rows)

    csv_summary = os.path.join(output_dir, "results.csv")
    with open(csv_summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "noise_level", "n",
            "cer_blur", "cer_wiener", "cer_gain",
            "wer_blur", "wer_wiener", "wer_gain",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary saved to {csv_summary}")

    csv_per = os.path.join(output_dir, "per_image.csv")
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
        plt.plot(xs, [float(r["cer_wiener"]) for r in summary_rows],
                 marker="s", label=f"Wiener restored (k={k})")
        plt.xlabel("Noise level (n_XX index)")
        plt.ylabel("Character Error Rate")
        plt.title("OCR test: CER vs noise level (lower is better)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        p1 = os.path.join(output_dir, "cer_vs_noise.png")
        plt.savefig(p1, dpi=150)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(xs, [float(r["wer_blur"]) for r in summary_rows],
                 marker="o", label="Blurred input (baseline)")
        plt.plot(xs, [float(r["wer_wiener"]) for r in summary_rows],
                 marker="s", label=f"Wiener restored (k={k})")
        plt.xlabel("Noise level (n_XX index)")
        plt.ylabel("Word Error Rate")
        plt.title("OCR test: WER vs noise level (lower is better)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        p2 = os.path.join(output_dir, "wer_vs_noise.png")
        plt.savefig(p2, dpi=150)
        plt.close()

        print(f"Plots saved to {p1} and {p2}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="data/BMVC_OCR_test_data")
    p.add_argument("--psf-dir", default=None, help="Defaults to <root>/psf")
    p.add_argument("--orig-dir", default=None, help="Defaults to <root>/orig")
    p.add_argument("--k", type=float, default=None)
    p.add_argument("--k-summary", default=None,
                   help="Path to tune_k_summary.json (overrides --k if --k not set)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap images per noise level (for quick runs)")
    p.add_argument("--noise-levels", default=None,
                   help="Comma-separated list of n_XX to restrict")
    p.add_argument("--psm", type=int, default=6)
    p.add_argument("--output-dir", default="classical/outputs/eval_ocr")
    a = p.parse_args()

    psf_dir = a.psf_dir or os.path.join(a.root, "psf")
    orig_dir = a.orig_dir or os.path.join(a.root, "orig")
    filt = set(x.strip() for x in a.noise_levels.split(",")) if a.noise_levels else None

    k = resolve_k(a)
    run(a.root, psf_dir, orig_dir, k, a.output_dir, a.limit, filt, a.psm)


if __name__ == "__main__":
    main()
