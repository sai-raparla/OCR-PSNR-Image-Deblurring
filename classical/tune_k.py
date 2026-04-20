"""Sweep the Wiener regularization constant k on a BMVC_image_data subset and
pick the value that minimizes mean character error rate (CER).

Usage (from repo root):
    python3 classical/tune_k.py --limit 200

Writes:
    classical/outputs/tuning/tune_k_cer.csv       per-(k, image) rows
    classical/outputs/tuning/tune_k_summary.json  {best_k, per_k_summary, ...}
"""

import argparse
import csv
import json
import os
import sys
import time
import numpy as np
import pytesseract
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "ocr"))

from wienerfiltering import open_pic
from metrics import match_ids_3way, wiener_restore_arr
from tesseract_eval import character_error_rate, run_ocr


def arr_to_pil(arr: np.ndarray) -> Image.Image:
    a = np.clip(arr, 0.0, 1.0) * 255.0
    return Image.fromarray(a.astype(np.uint8))


def ocr_pil(img: Image.Image, psm: int) -> str:
    return pytesseract.image_to_string(img, config=f"--psm {psm}").strip()


def parse_k_list(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def run_tuning(blur_dir, psf_dir, orig_dir, output_dir, limit, k_list, psm):
    os.makedirs(output_dir, exist_ok=True)

    ids, blur_map, psf_map, orig_map = match_ids_3way(blur_dir, psf_dir, orig_dir)
    if not ids:
        print("ERROR: no IDs common to blur, psf, and orig folders.")
        sys.exit(1)

    if limit is not None:
        ids = ids[:limit]

    n = len(ids)
    print(f"Tuning on {n} images, k values = {k_list}")
    print(f"Tesseract PSM: {psm}")
    print("-" * 70)

    print(f"Step 1/2: OCR reference texts from orig/ ({n} images)")
    orig_texts = {}
    t0 = time.time()
    for i, img_id in enumerate(ids, 1):
        orig_path = os.path.join(orig_dir, orig_map[img_id])
        try:
            orig_texts[img_id] = run_ocr(orig_path, psm)["text"]
        except Exception as e:
            orig_texts[img_id] = ""
            print(f"  WARN: failed to OCR orig {img_id}: {e}")
        if i % 20 == 0 or i == n:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (n - i) / rate if rate > 0 else 0.0
            print(f"  [{i}/{n}] {rate:.2f} img/s | ETA: {eta:.0f}s")

    blur_cache = {}
    psf_cache = {}
    print(f"\nStep 2/2: sweep {len(k_list)} k values x {n} images ="
          f" {len(k_list) * n} Wiener + OCR ops")

    per_row = []
    per_k_summary = []

    for k_idx, k in enumerate(k_list, 1):
        print(f"\n--- k = {k} ({k_idx}/{len(k_list)}) ---")
        cer_list = []
        blank_count = 0
        t0 = time.time()

        for i, img_id in enumerate(ids, 1):
            blur_path = os.path.join(blur_dir, blur_map[img_id])
            psf_path = os.path.join(psf_dir, psf_map[img_id])

            try:
                if img_id not in blur_cache:
                    blur_cache[img_id] = open_pic(blur_path)
                if img_id not in psf_cache:
                    psf_cache[img_id] = open_pic(psf_path)

                blur_arr = blur_cache[img_id]
                psf_arr = psf_cache[img_id]

                restored = wiener_restore_arr(blur_arr, psf_arr, k)
                pil_restored = arr_to_pil(restored)
                hyp = ocr_pil(pil_restored, psm)

                ref = orig_texts.get(img_id, "")
                cer = character_error_rate(ref, hyp)
            except Exception as e:
                print(f"  WARN: id={img_id} k={k} error: {e}")
                continue

            if len(hyp) == 0:
                blank_count += 1

            cer_list.append(cer)
            per_row.append({
                "k": f"{k:.6g}",
                "image_id": img_id,
                "cer_wiener": f"{cer:.4f}",
                "len_orig": len(ref),
                "len_wiener": len(hyp),
            })

            if i % 20 == 0 or i == n:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (n - i) / rate if rate > 0 else 0.0
                print(f"  [{i}/{n}] {rate:.2f} img/s | ETA: {eta:.0f}s")

        if cer_list:
            mean_cer = float(np.mean(cer_list))
            median_cer = float(np.median(cer_list))
        else:
            mean_cer = float("inf")
            median_cer = float("inf")

        per_k_summary.append({
            "k": k,
            "n": len(cer_list),
            "mean_cer": mean_cer,
            "median_cer": median_cer,
            "blank_outputs": blank_count,
        })
        print(f"  k={k}: mean CER={mean_cer:.4f}, median CER={median_cer:.4f},"
              f" blanks={blank_count}/{len(cer_list)}")

    per_k_summary.sort(key=lambda r: r["mean_cer"])
    best = per_k_summary[0]

    print("\n" + "=" * 70)
    print("TUNING SUMMARY (sorted by mean CER, lower is better)")
    print("=" * 70)
    print(f"  {'k':>10}  {'mean CER':>10}  {'median CER':>12}  {'blanks':>8}  {'n':>5}")
    for row in per_k_summary:
        print(f"  {row['k']:>10.4g}  {row['mean_cer']:>10.4f}"
              f"  {row['median_cer']:>12.4f}  {row['blank_outputs']:>8d}  {row['n']:>5d}")
    print("-" * 70)
    print(f"Best k = {best['k']} (mean CER = {best['mean_cer']:.4f})")
    print("=" * 70)

    csv_path = os.path.join(output_dir, "tune_k_cer.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["k", "image_id", "cer_wiener", "len_orig", "len_wiener"]
        )
        writer.writeheader()
        writer.writerows(per_row)
    print(f"\nPer-(k, image) rows saved to: {csv_path}")

    summary_path = os.path.join(output_dir, "tune_k_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "best_k": best["k"],
                "best_mean_cer": best["mean_cer"],
                "n_images": n,
                "psm": psm,
                "k_list": k_list,
                "per_k_summary": per_k_summary,
                "blur_dir": blur_dir,
                "psf_dir": psf_dir,
                "orig_dir": orig_dir,
            },
            f,
            indent=2,
        )
    print(f"Summary (with best_k) saved to: {summary_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--blur-dir", default="data/BMVC_image_data/blur")
    p.add_argument("--psf-dir", default="data/BMVC_image_data/psf")
    p.add_argument("--orig-dir", default="data/BMVC_image_data/orig")
    p.add_argument("--limit", type=int, default=200,
                   help="Cap on images per k sweep (default: 200)")
    p.add_argument("--k-list", type=parse_k_list,
                   default=parse_k_list("0.001,0.005,0.01,0.02,0.05,0.1"),
                   help="Comma-separated k values")
    p.add_argument("--psm", type=int, default=6)
    p.add_argument("--output-dir", default="classical/outputs/tuning")

    a = p.parse_args()
    run_tuning(a.blur_dir, a.psf_dir, a.orig_dir, a.output_dir,
               a.limit, a.k_list, a.psm)


if __name__ == "__main__":
    main()
