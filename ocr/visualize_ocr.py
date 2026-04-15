"""
Run Tesseract OCR on BMVC_image_data (blur vs orig) and visualize performance.
"""

import argparse
import os
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tesseract_eval import find_paired_images, run_ocr, character_error_rate, word_error_rate, extract_image_id


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "BMVC_image_data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def run_evaluation(blur_dir, orig_dir, limit=None, psm=6):
    pairs = find_paired_images(blur_dir, orig_dir)
    if limit:
        pairs = pairs[:limit]

    total = len(pairs)
    print(f"Found {total} matched blur/orig image pairs")
    print(f"Tesseract PSM mode: {psm}")
    print("-" * 60)

    results = []
    skipped = 0
    t0 = time.time()

    for i, (blur_fname, orig_fname) in enumerate(pairs):
        try:
            orig_result = run_ocr(os.path.join(orig_dir, orig_fname), psm)
            blur_result = run_ocr(os.path.join(blur_dir, blur_fname), psm)
        except Exception:
            skipped += 1
            continue

        ref = orig_result["text"]
        hyp = blur_result["text"]
        cer = character_error_rate(ref, hyp)
        wer = word_error_rate(ref, hyp)

        results.append({
            "image_id": extract_image_id(blur_fname),
            "cer": cer,
            "wer": wer,
            "conf_orig": orig_result["avg_confidence"],
            "conf_blur": blur_result["avg_confidence"],
            "orig_text_len": len(ref),
            "blur_text_len": len(hyp),
        })

        processed = len(results) + skipped
        if processed % 10 == 0 or processed == total:
            elapsed = time.time() - t0
            rate = processed / elapsed
            eta = (total - processed) / rate if rate > 0 else 0
            print(f"  [{processed}/{total}] {rate:.1f} img/s | ETA: {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone: {len(results)} evaluated, {skipped} skipped, {elapsed:.1f}s total")
    return results


def plot_results(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n = len(results)

    cers = [r["cer"] * 100 for r in results]
    wers = [r["wer"] * 100 for r in results]
    conf_origs = [r["conf_orig"] for r in results]
    conf_blurs = [r["conf_blur"] for r in results]
    ids = list(range(n))

    avg_cer = np.mean(cers)
    avg_wer = np.mean(wers)
    avg_conf_orig = np.mean(conf_origs)
    avg_conf_blur = np.mean(conf_blurs)
    exact_match_pct = sum(1 for c in cers if c == 0) / n * 100

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Tesseract OCR: Blurred vs Original (BMVC_image_data, n={n})",
        fontsize=14, fontweight="bold",
    )

    # 1. CER distribution histogram
    ax = axes[0, 0]
    ax.hist(cers, bins=30, color="#e74c3c", edgecolor="white", alpha=0.85)
    ax.axvline(avg_cer, color="#c0392b", linestyle="--", linewidth=2, label=f"Mean CER = {avg_cer:.1f}%")
    ax.set_xlabel("Character Error Rate (%)")
    ax.set_ylabel("Number of Images")
    ax.set_title("CER Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 2. WER distribution histogram
    ax = axes[0, 1]
    ax.hist(wers, bins=30, color="#3498db", edgecolor="white", alpha=0.85)
    ax.axvline(avg_wer, color="#2471a3", linestyle="--", linewidth=2, label=f"Mean WER = {avg_wer:.1f}%")
    ax.set_xlabel("Word Error Rate (%)")
    ax.set_ylabel("Number of Images")
    ax.set_title("WER Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Confidence: orig vs blur scatter
    ax = axes[1, 0]
    ax.scatter(conf_origs, conf_blurs, alpha=0.5, s=20, color="#8e44ad", edgecolors="none")
    lims = [0, 100]
    ax.plot(lims, lims, "--", color="#bdc3c7", linewidth=1, label="No degradation")
    ax.set_xlabel("Original Confidence")
    ax.set_ylabel("Blurred Confidence")
    ax.set_title("Confidence: Original vs Blurred")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # 4. Summary stats as a text panel
    ax = axes[1, 1]
    ax.axis("off")
    stats_text = (
        f"Images Evaluated:   {n}\n\n"
        f"Avg CER:            {avg_cer:.1f}%\n"
        f"Avg WER:            {avg_wer:.1f}%\n\n"
        f"Avg Confidence (orig):  {avg_conf_orig:.1f}\n"
        f"Avg Confidence (blur):  {avg_conf_blur:.1f}\n"
        f"Confidence Drop:        {avg_conf_orig - avg_conf_blur:.1f}\n\n"
        f"Exact OCR Matches:  {exact_match_pct:.1f}%\n"
        f"Median CER:         {np.median(cers):.1f}%\n"
        f"Median WER:         {np.median(wers):.1f}%\n"
    )
    ax.text(
        0.1, 0.5, stats_text,
        transform=ax.transAxes,
        fontsize=13, fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#ecf0f1", edgecolor="#bdc3c7"),
    )
    ax.set_title("Summary Statistics")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ocr_performance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize Tesseract OCR on BMVC_image_data blur vs orig"
    )
    parser.add_argument(
        "--blur-dir", default=os.path.join(DATA_DIR, "blur"),
        help="Path to blurred images",
    )
    parser.add_argument(
        "--orig-dir", default=os.path.join(DATA_DIR, "orig"),
        help="Path to original images",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images")
    parser.add_argument("--psm", type=int, default=6, help="Tesseract PSM mode")
    args = parser.parse_args()

    results = run_evaluation(args.blur_dir, args.orig_dir, args.limit, args.psm)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, "ocr_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Raw results saved to: {json_path}")

    plot_results(results)


if __name__ == "__main__":
    main()
