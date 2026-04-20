"""
Produce two comparison charts from a tesseract_eval.py CSV:
  1. confidence_comparison.png  - OCR confidence on original vs blurred images
  2. error_rates_comparison.png - CER and WER distributions on blurred images
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "outputs", "ocr_results_250.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def load_results(csv_path: str):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "image_id": r["image_id"],
                "orig_conf": float(r["orig_confidence"]),
                "blur_conf": float(r["blur_confidence"]),
                "cer": float(r["cer"]) * 100.0,
                "wer": float(r["wer"]) * 100.0,
            })
    return rows


def plot_confidence_comparison(rows, out_path: str):
    orig_conf = np.array([r["orig_conf"] for r in rows])
    blur_conf = np.array([r["blur_conf"] for r in rows])
    n = len(rows)

    fig, (ax_hist, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Tesseract OCR Confidence: Original vs Blurred (n={n})",
        fontsize=14, fontweight="bold",
    )

    bins = np.linspace(0, 100, 31)
    ax_hist.hist(orig_conf, bins=bins, color="#27ae60", alpha=0.65,
                 edgecolor="white", label=f"Original (mean={orig_conf.mean():.1f})")
    ax_hist.hist(blur_conf, bins=bins, color="#e74c3c", alpha=0.65,
                 edgecolor="white", label=f"Blurred  (mean={blur_conf.mean():.1f})")
    ax_hist.axvline(orig_conf.mean(), color="#1e8449", linestyle="--", linewidth=1.5)
    ax_hist.axvline(blur_conf.mean(), color="#922b21", linestyle="--", linewidth=1.5)
    ax_hist.set_xlabel("Tesseract confidence")
    ax_hist.set_ylabel("Number of images")
    ax_hist.set_title("Confidence distribution")
    ax_hist.legend(loc="upper left")
    ax_hist.grid(True, alpha=0.3, axis="y")

    labels = ["Original", "Blurred"]
    means = [orig_conf.mean(), blur_conf.mean()]
    medians = [np.median(orig_conf), np.median(blur_conf)]
    x = np.arange(len(labels))
    width = 0.35
    ax_bar.bar(x - width / 2, means, width, color=["#27ae60", "#e74c3c"],
               label="Mean", edgecolor="white")
    ax_bar.bar(x + width / 2, medians, width, color=["#27ae60", "#e74c3c"],
               alpha=0.55, label="Median", edgecolor="white", hatch="//")
    for i, (m, md) in enumerate(zip(means, medians)):
        ax_bar.text(i - width / 2, m + 1, f"{m:.1f}", ha="center", fontsize=10)
        ax_bar.text(i + width / 2, md + 1, f"{md:.1f}", ha="center", fontsize=10)
    drop = orig_conf.mean() - blur_conf.mean()
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylim(0, 100)
    ax_bar.set_ylabel("Confidence")
    ax_bar.set_title(f"Mean / Median confidence  (drop = {drop:.1f})")
    ax_bar.legend()
    ax_bar.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_error_rates_comparison(rows, out_path: str):
    cer = np.array([r["cer"] for r in rows])
    wer = np.array([r["wer"] for r in rows])
    n = len(rows)

    fig, (ax_hist, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"OCR Error on Blurred Images vs Original (n={n})",
        fontsize=14, fontweight="bold",
    )

    bins = np.linspace(0, max(cer.max(), wer.max(), 100), 31)
    ax_hist.hist(cer, bins=bins, color="#e67e22", alpha=0.65,
                 edgecolor="white", label=f"CER (mean={cer.mean():.1f}%)")
    ax_hist.hist(wer, bins=bins, color="#2980b9", alpha=0.65,
                 edgecolor="white", label=f"WER (mean={wer.mean():.1f}%)")
    ax_hist.axvline(cer.mean(), color="#a04000", linestyle="--", linewidth=1.5)
    ax_hist.axvline(wer.mean(), color="#1b4f72", linestyle="--", linewidth=1.5)
    ax_hist.set_xlabel("Error rate (%)")
    ax_hist.set_ylabel("Number of images")
    ax_hist.set_title("CER and WER distribution")
    ax_hist.legend(loc="upper right")
    ax_hist.grid(True, alpha=0.3, axis="y")

    labels = ["CER", "WER"]
    means = [cer.mean(), wer.mean()]
    medians = [np.median(cer), np.median(wer)]
    exact = float((cer == 0).sum()) / n * 100.0
    x = np.arange(len(labels))
    width = 0.35
    ax_bar.bar(x - width / 2, means, width, color=["#e67e22", "#2980b9"],
               label="Mean", edgecolor="white")
    ax_bar.bar(x + width / 2, medians, width, color=["#e67e22", "#2980b9"],
               alpha=0.55, label="Median", edgecolor="white", hatch="//")
    for i, (m, md) in enumerate(zip(means, medians)):
        ax_bar.text(i - width / 2, m + 1, f"{m:.1f}%", ha="center", fontsize=10)
        ax_bar.text(i + width / 2, md + 1, f"{md:.1f}%", ha="center", fontsize=10)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylabel("Error rate (%)")
    ax_bar.set_title(f"Mean / Median error  (exact matches = {exact:.1f}%)")
    ax_bar.legend()
    ax_bar.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build two comparison charts from a tesseract_eval CSV"
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to tesseract_eval CSV")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory to save charts")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(
            f"CSV not found: {args.csv}\n"
            f"Run: python ocr/tesseract_eval.py --limit 250 "
            f"--output-csv {args.csv}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_results(args.csv)
    print(f"Loaded {len(rows)} rows from {args.csv}")

    plot_confidence_comparison(
        rows, os.path.join(args.output_dir, "confidence_comparison.png")
    )
    plot_error_rates_comparison(
        rows, os.path.join(args.output_dir, "error_rates_comparison.png")
    )


if __name__ == "__main__":
    main()
