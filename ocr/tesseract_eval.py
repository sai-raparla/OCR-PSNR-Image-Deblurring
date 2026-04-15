
import argparse
import os
import re
import sys
import csv
import time
import pytesseract
from PIL import Image


def compute_edit_distance(ref: str, hyp: str) -> int:
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def character_error_rate(reference: str, hypothesis: str) -> float:
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return compute_edit_distance(reference, hypothesis) / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return compute_edit_distance(ref_words, hyp_words) / len(ref_words)


def extract_image_id(filename: str) -> str:
    match = re.match(r"(\d+)", filename)
    return match.group(1) if match else None


def run_ocr(image_path: str, psm: int = 6) -> dict:
    """Run Tesseract on a single image, return text and confidence data."""
    img = Image.open(image_path)
    custom_config = f"--psm {psm}"
    text = pytesseract.image_to_string(img, config=custom_config).strip()
    data = pytesseract.image_to_data(img, config=custom_config, output_type=pytesseract.Output.DICT)
    confidences = [int(c) for c in data["conf"] if int(c) > 0]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return {"text": text, "avg_confidence": avg_conf, "num_words": len(confidences)}


def find_paired_images(blur_dir: str, orig_dir: str):
    """Find blur images that have a matching original by numeric ID."""
    blur_files = {
        extract_image_id(f): f
        for f in os.listdir(blur_dir)
        if f.endswith(".png") and extract_image_id(f) is not None
    }
    orig_files = {
        extract_image_id(f): f
        for f in os.listdir(orig_dir)
        if f.endswith(".png") and extract_image_id(f) is not None
    }
    common_ids = sorted(set(blur_files) & set(orig_files))
    pairs = [(blur_files[id_], orig_files[id_]) for id_ in common_ids]
    return pairs


def evaluate(blur_dir, orig_dir, limit, psm, output_csv):
    pairs = find_paired_images(blur_dir, orig_dir)
    if limit:
        pairs = pairs[:limit]

    total = len(pairs)
    if total == 0:
        print("ERROR: No matching blur/orig image pairs found.")
        print(f"  blur dir: {blur_dir}")
        print(f"  orig dir: {orig_dir}")
        sys.exit(1)

    print(f"Found {total} matched blur/orig image pairs")
    print(f"Tesseract PSM mode: {psm}")
    print("-" * 70)

    results = []
    total_cer_blur = 0.0
    total_cer_orig = 0.0
    total_wer_blur = 0.0
    total_wer_orig = 0.0
    total_conf_blur = 0.0
    total_conf_orig = 0.0
    orig_texts = []
    blur_texts = []

    t0 = time.time()

    skipped = 0
    for i, (blur_fname, orig_fname) in enumerate(pairs):
        blur_path = os.path.join(blur_dir, blur_fname)
        orig_path = os.path.join(orig_dir, orig_fname)

        try:
            orig_result = run_ocr(orig_path, psm)
            blur_result = run_ocr(blur_path, psm)
        except Exception as e:
            skipped += 1
            continue

        ref_text = orig_result["text"]
        hyp_text = blur_result["text"]

        cer = character_error_rate(ref_text, hyp_text)
        wer = word_error_rate(ref_text, hyp_text)

        total_cer_blur += cer
        total_wer_blur += wer
        total_conf_blur += blur_result["avg_confidence"]
        total_conf_orig += orig_result["avg_confidence"]

        orig_texts.append(ref_text)
        blur_texts.append(hyp_text)

        row = {
            "image_id": extract_image_id(blur_fname),
            "orig_file": orig_fname,
            "blur_file": blur_fname,
            "orig_text": ref_text[:80],
            "blur_text": hyp_text[:80],
            "orig_confidence": f"{orig_result['avg_confidence']:.1f}",
            "blur_confidence": f"{blur_result['avg_confidence']:.1f}",
            "cer": f"{cer:.4f}",
            "wer": f"{wer:.4f}",
        }
        results.append(row)

        processed = len(results) + skipped
        if processed % 20 == 0 or processed == total:
            elapsed = time.time() - t0
            rate = processed / elapsed
            eta = (total - processed) / rate if rate > 0 else 0
            print(f"  [{processed}/{total}] {rate:.1f} img/s | ETA: {eta:.0f}s")

    elapsed = time.time() - t0

    evaluated = total - skipped
    if evaluated == 0:
        print("ERROR: All images failed to process.")
        sys.exit(1)

    # Aggregate metrics
    avg_cer = total_cer_blur / evaluated
    avg_wer = total_wer_blur / evaluated
    avg_conf_orig = total_conf_orig / evaluated
    avg_conf_blur = total_conf_blur / evaluated

    # Corpus-level CER: concatenate all texts
    all_orig = " ".join(orig_texts)
    all_blur = " ".join(blur_texts)
    corpus_cer = character_error_rate(all_orig, all_blur) if all_orig else 0.0
    corpus_wer = word_error_rate(all_orig, all_blur) if all_orig else 0.0

    exact_match = sum(1 for r in results if float(r["cer"]) == 0.0)
    no_text_blur = sum(1 for t in blur_texts if len(t.strip()) == 0)
    no_text_orig = sum(1 for t in orig_texts if len(t.strip()) == 0)

    print("\n" + "=" * 70)
    print("OCR PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"  Images evaluated:         {evaluated} (skipped {skipped} corrupt)")
    print(f"  Time elapsed:             {elapsed:.1f}s ({evaluated/elapsed:.1f} img/s)")
    print()
    print(f"  Avg CER (per-image):      {avg_cer:.4f}  ({avg_cer*100:.2f}%)")
    print(f"  Avg WER (per-image):      {avg_wer:.4f}  ({avg_wer*100:.2f}%)")
    print(f"  Corpus CER:               {corpus_cer:.4f}  ({corpus_cer*100:.2f}%)")
    print(f"  Corpus WER:               {corpus_wer:.4f}  ({corpus_wer*100:.2f}%)")
    print()
    print(f"  Avg confidence (orig):    {avg_conf_orig:.1f}")
    print(f"  Avg confidence (blur):    {avg_conf_blur:.1f}")
    print(f"  Confidence drop:          {avg_conf_orig - avg_conf_blur:.1f}")
    print()
    print(f"  Exact OCR matches:        {exact_match}/{evaluated} ({exact_match/evaluated*100:.1f}%)")
    print(f"  Blank OCR output (orig):  {no_text_orig}/{evaluated}")
    print(f"  Blank OCR output (blur):  {no_text_blur}/{evaluated}")
    print("=" * 70)

    if output_csv:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nPer-image results saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Tesseract OCR on blurred vs original image pairs"
    )
    parser.add_argument(
        "--blur-dir",
        default="data/BMVC_image_data/blur",
        help="Path to folder of blurred images (default: data/BMVC_image_data/blur)",
    )
    parser.add_argument(
        "--orig-dir",
        default="data/BMVC_image_data/orig",
        help="Path to folder of original images (default: data/BMVC_image_data/orig)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process (useful for quick tests)",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=6,
        help="Tesseract page segmentation mode (default: 6 = uniform block of text)",
    )
    parser.add_argument(
        "--output-csv",
        default="ocr/outputs/ocr_results.csv",
        help="Path to save per-image CSV results",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    evaluate(args.blur_dir, args.orig_dir, args.limit, args.psm, args.output_csv)


if __name__ == "__main__":
    main()
