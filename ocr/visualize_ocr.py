"""
Run Tesseract OCR across all blur levels in BMVC_OCR_test_data
and produce performance visualizations.
"""

import os
import sys
import json
import time

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from tesseract_eval import find_paired_images, run_ocr, character_error_rate, word_error_rate, extract_image_id


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "BMVC_OCR_test_data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def sweep_blur_levels(limit=None, psm=6):
    orig_dir = os.path.join(DATA_DIR, "orig")
    blur_levels = sorted(
        d for d in os.listdir(DATA_DIR)
        if d.startswith("n_") and os.path.isdir(os.path.join(DATA_DIR, d))
    )

    print(f"Found {len(blur_levels)} blur levels: {blur_levels}")
    print(f"Orig dir: {orig_dir}")
    if limit:
        print(f"Limit: {limit} images per level")
    print()

    all_results = {}

    for level in blur_levels:
        blur_dir = os.path.join(DATA_DIR, level)
        pairs = find_paired_images(blur_dir, orig_dir)
        if limit:
            pairs = pairs[:limit]

        cers, wers, conf_origs, conf_blurs = [], [], [], []
        skipped = 0
        t0 = time.time()

        for blur_fname, orig_fname in pairs:
            try:
                orig_result = run_ocr(os.path.join(orig_dir, orig_fname), psm)
                blur_result = run_ocr(os.path.join(blur_dir, blur_fname), psm)
            except Exception:
                skipped += 1
                continue

            ref = orig_result["text"]
            hyp = blur_result["text"]
            cers.append(character_error_rate(ref, hyp))
            wers.append(word_error_rate(ref, hyp))
            conf_origs.append(orig_result["avg_confidence"])
            conf_blurs.append(blur_result["avg_confidence"])

        elapsed = time.time() - t0
        n = len(cers)
        if n == 0:
            print(f"  {level}: all images failed, skipping")
            continue

        result = {
            "level": level,
            "num_images": n,
            "skipped": skipped,
            "avg_cer": sum(cers) / n,
            "avg_wer": sum(wers) / n,
            "avg_conf_orig": sum(conf_origs) / n,
            "avg_conf_blur": sum(conf_blurs) / n,
            "exact_matches": sum(1 for c in cers if c == 0.0),
        }
        all_results[level] = result
        print(
            f"  {level}: CER={result['avg_cer']:.3f}  WER={result['avg_wer']:.3f}  "
            f"Conf={result['avg_conf_blur']:.1f}  ({n} imgs, {elapsed:.1f}s)"
        )

    return all_results


def plot_results(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    levels = sorted(results.keys())
    level_nums = [int(l.split("_")[1]) for l in levels]
    cers = [results[l]["avg_cer"] * 100 for l in levels]
    wers = [results[l]["avg_wer"] * 100 for l in levels]
    conf_orig = [results[l]["avg_conf_orig"] for l in levels]
    conf_blur = [results[l]["avg_conf_blur"] for l in levels]
    exact = [results[l]["exact_matches"] / results[l]["num_images"] * 100 for l in levels]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Tesseract OCR Performance vs Blur Level (BMVC_OCR_test_data)", fontsize=14, fontweight="bold")

    # CER & WER
    ax = axes[0, 0]
    ax.plot(level_nums, cers, "o-", color="#e74c3c", linewidth=2, markersize=7, label="CER")
    ax.plot(level_nums, wers, "s-", color="#3498db", linewidth=2, markersize=7, label="WER")
    ax.set_xlabel("Blur Level")
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("Character & Word Error Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(level_nums)

    # Confidence
    ax = axes[0, 1]
    ax.plot(level_nums, conf_orig, "o-", color="#27ae60", linewidth=2, markersize=7, label="Original")
    ax.plot(level_nums, conf_blur, "s-", color="#e67e22", linewidth=2, markersize=7, label="Blurred")
    ax.fill_between(level_nums, conf_blur, conf_orig, alpha=0.15, color="#e67e22")
    ax.set_xlabel("Blur Level")
    ax.set_ylabel("Avg Confidence")
    ax.set_title("Tesseract Confidence Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(level_nums)

    # Confidence drop
    ax = axes[1, 0]
    drops = [o - b for o, b in zip(conf_orig, conf_blur)]
    colors = ["#2ecc71" if d < 20 else "#f39c12" if d < 40 else "#e74c3c" for d in drops]
    ax.bar(level_nums, drops, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Blur Level")
    ax.set_ylabel("Confidence Drop")
    ax.set_title("Confidence Loss (Original - Blurred)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(level_nums)

    # Exact match rate
    ax = axes[1, 1]
    ax.bar(level_nums, exact, color="#9b59b6", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Blur Level")
    ax.set_ylabel("Exact Match (%)")
    ax.set_title("Images with Identical OCR Output")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(level_nums)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ocr_performance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sweep blur levels and visualize OCR performance")
    parser.add_argument("--limit", type=int, default=None, help="Limit images per blur level")
    parser.add_argument("--psm", type=int, default=6, help="Tesseract PSM mode")
    args = parser.parse_args()

    results = sweep_blur_levels(limit=args.limit, psm=args.psm)

    json_path = os.path.join(OUTPUT_DIR, "sweep_results.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to: {json_path}")

    plot_results(results)


if __name__ == "__main__":
    main()
