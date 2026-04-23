"""Compare Wiener vs Restormer head-to-head.

Reads the summary CSVs produced by:
    classical/outputs/eval_iq/results.csv
    classical/outputs/eval_ocr/results.csv
    restormer/outputs/eval_iq/results.csv
    restormer/outputs/eval_ocr/results.csv

and produces combined plots + a merged CSV showing the absolute numbers plus
the improvement of Restormer over Wiener.

Usage (from repo root):
    python3 restormer/compare.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def noise_idx(level: str) -> int:
    return int(level.split("_")[1])


def join_by_level(rows_a: list, rows_b: list, key_a: str, key_b: str):
    """Inner-join two row lists on `noise_level`, return dict keyed by level."""
    a = {r["noise_level"]: r for r in rows_a}
    b = {r["noise_level"]: r for r in rows_b}
    levels = sorted(set(a) & set(b), key=noise_idx)
    out = {}
    for lvl in levels:
        out[lvl] = {
            "noise_level": lvl,
            "baseline": float(a[lvl][key_a.replace("_wiener", "_blur").replace("_restormer", "_blur")]),
            "wiener": float(a[lvl][key_a]),
            "restormer": float(b[lvl][key_b]),
        }
    return out


def plot_compare(joined, metric_name, ylabel, title, out_path, lower_is_better):
    if not joined:
        print(f"  (no data for {metric_name})"); return
    xs = [noise_idx(l) for l in joined]
    base = [joined[l]["baseline"] for l in joined]
    wien = [joined[l]["wiener"] for l in joined]
    rest = [joined[l]["restormer"] for l in joined]

    plt.figure(figsize=(9, 5.5))
    plt.plot(xs, base, marker="o", linestyle="--", color="#888",
             label="Blurred input (baseline)")
    plt.plot(xs, wien, marker="s", color="#d1622d", label="Wiener")
    plt.plot(xs, rest, marker="^", color="#2a7fbf", label="Restormer")
    plt.xlabel("Noise level (n_XX index)")
    plt.ylabel(ylabel)
    suffix = " (lower is better)" if lower_is_better else " (higher is better)"
    plt.title(title + suffix)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  plot -> {out_path}")


def write_merged_csv(joined, out_path, metric_key):
    if not joined:
        return
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "noise_level", f"{metric_key}_blur", f"{metric_key}_wiener",
            f"{metric_key}_restormer",
            f"wiener_gain_vs_blur", f"restormer_gain_vs_blur",
            f"restormer_vs_wiener",
        ])
        for lvl, d in joined.items():
            base = d["baseline"]
            wien = d["wiener"]
            rest = d["restormer"]
            w.writerow([
                lvl, f"{base:.4f}", f"{wien:.4f}", f"{rest:.4f}",
                f"{wien - base:+.4f}", f"{rest - base:+.4f}",
                f"{rest - wien:+.4f}",
            ])
    print(f"  csv  -> {out_path}")


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    wiener_iq = read_csv(args.wiener_iq)
    wiener_ocr = read_csv(args.wiener_ocr)
    restormer_iq = read_csv(args.restormer_iq)
    restormer_ocr = read_csv(args.restormer_ocr)

    any_data = False

    # PSNR
    if wiener_iq and restormer_iq:
        any_data = True
        print("[PSNR] Wiener vs Restormer")
        joined = join_by_level(wiener_iq, restormer_iq, "psnr_wiener", "psnr_restormer")
        plot_compare(
            joined, "psnr", "PSNR (dB)",
            "Image-quality test: PSNR vs noise level",
            os.path.join(args.output_dir, "psnr_compare.png"),
            lower_is_better=False,
        )
        write_merged_csv(joined, os.path.join(args.output_dir, "psnr_compare.csv"), "psnr")
    else:
        print("[PSNR] Skipping (missing wiener or restormer IQ CSV)")

    # CER
    if wiener_ocr and restormer_ocr:
        any_data = True
        print("\n[CER] Wiener vs Restormer")
        joined = join_by_level(wiener_ocr, restormer_ocr, "cer_wiener", "cer_restormer")
        plot_compare(
            joined, "cer", "Character Error Rate",
            "OCR test: CER vs noise level",
            os.path.join(args.output_dir, "cer_compare.png"),
            lower_is_better=True,
        )
        write_merged_csv(joined, os.path.join(args.output_dir, "cer_compare.csv"), "cer")

        # WER
        print("\n[WER] Wiener vs Restormer")
        joined = join_by_level(wiener_ocr, restormer_ocr, "wer_wiener", "wer_restormer")
        plot_compare(
            joined, "wer", "Word Error Rate",
            "OCR test: WER vs noise level",
            os.path.join(args.output_dir, "wer_compare.png"),
            lower_is_better=True,
        )
        write_merged_csv(joined, os.path.join(args.output_dir, "wer_compare.csv"), "wer")
    else:
        print("[CER/WER] Skipping (missing wiener or restormer OCR CSV)")

    if not any_data:
        print("No comparable CSVs found; run both classical and restormer evals first.")
        sys.exit(1)

    print(f"\nAll comparisons written to {args.output_dir}/")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--wiener-iq", default="classical/outputs/eval_iq/results.csv")
    p.add_argument("--wiener-ocr", default="classical/outputs/eval_ocr/results.csv")
    p.add_argument("--restormer-iq", default="restormer/outputs/eval_iq/results.csv")
    p.add_argument("--restormer-ocr", default="restormer/outputs/eval_ocr/results.csv")
    p.add_argument("--output-dir", default="restormer/outputs/compare")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
