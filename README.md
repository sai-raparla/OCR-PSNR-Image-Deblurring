# OCR-Image-Deblurring

Deblurring text images to improve OCR accuracy using computer vision techniques.

Classical Wiener deconvolution is used as the baseline restoration method. The pipeline tunes the Wiener regularization constant `k` on a training subset, then evaluates the locked-in `k` on two held-out test sets: one for image-quality metrics (PSNR) and one for OCR metrics (CER, WER).

Dataset: [Kaggle — Text Deblurring Dataset with PSF for OCR](https://www.kaggle.com/datasets/anggadwisunarto/text-deblurring-dataset-with-psf-for-ocr).

## Repository layout

```
OCR-Image-Deblurring/
├── classical/
│   ├── wienerfiltering.py       core Wiener deconvolution (FFT-based) + CLI batch driver
│   ├── metrics.py               shared helpers: psnr(), 3-way ID match, wiener_restore_arr()
│   ├── tune_k.py                sweep k on a training subset, pick best by CER
│   ├── eval_iq.py               evaluate locked k with PSNR on BMVC_image_quality_test_data
│   ├── eval_ocr.py              evaluate locked k with CER/WER on BMVC_OCR_test_data
│   ├── make_preview_train.py    produce a Blur | Original | Wiener Restored PNG
│   ├── check_one.py             tiny sanity probe
│   └── outputs/                 generated results (CSVs, plots, restored PNGs)
├── ocr/
│   ├── tesseract_eval.py        run Tesseract on blur vs orig pairs, compute CER/WER
│   └── visualize_ocr.py         visualize OCR results CSV
├── data/
│   ├── BMVC_image_data/                  training pool: blur/, psf/, orig/
│   ├── BMVC_image_quality_test_data/     IQ test set: 200x200, n_00..n_15, 100 images/level
│   └── BMVC_OCR_test_data/               OCR test set: 512x512, n_00..n_10, 94 images/level
├── requirements.txt
└── README.md
```

## Installation

Requires Python 3.10+ and a local Tesseract binary (`brew install tesseract` on macOS, `apt install tesseract-ocr` on Debian/Ubuntu).

```bash
pip install -r requirements.txt
```

Python deps: `numpy`, `Pillow`, `pytesseract`, `matplotlib`.

## Datasets

All three folders share the same triplet structure (blur image, PSF kernel, ground-truth orig), but they differ in purpose:

| Folder | Role | Image size | Noise levels | Images |
|---|---|---|---|---|
| `BMVC_image_data/` | Training / tuning pool | Variable | none (one blur level) | ~44k triplets |
| `BMVC_image_quality_test_data/` | Image-quality test set | 200x200 | `n_00`..`n_15` (16) | 100 per level |
| `BMVC_OCR_test_data/` | OCR test set | 512x512 | `n_00`..`n_10` (11) | 94 per level |

The IQ test set is small so PSNR sweeps over 16 noise levels run fast. The OCR test set is larger because Tesseract needs roughly 20+ pixels per character for reliable recognition.

Note: in `BMVC_image_data/`, the `blur/` and `psf/` folders start at ID `0000000` but `orig/` starts at ID `0009059`. The tuning script uses the sorted 3-way intersection of IDs (22,344 common IDs).

## Pipeline: tune, then evaluate

The classical pipeline has three stages. They must be run in this order the first time (steps 2 and 3 read the tuned `k` from step 1):

### 1. Tune `k` on the training subset

```bash
python3 classical/tune_k.py --limit 200
```

Sweeps the Wiener regularization constant `k` over `[0.001, 0.005, 0.01, 0.02, 0.05, 0.1]` on 200 training images. For each `k`, deblurs every image, runs Tesseract on the result, and computes CER against the OCR of the ground-truth original. Picks the `k` that minimizes mean CER.

Outputs:
- `classical/outputs/tuning/tune_k_cer.csv` — one row per (k, image).
- `classical/outputs/tuning/tune_k_summary.json` — contains `best_k`, consumed by the next two scripts.

Cost: ~25 minutes (Tesseract-dominated). Useful flags:
- `--limit N` — images per sweep (default 200).
- `--k-list 0.005,0.01,0.02` — custom k grid.
- `--psm 6` — Tesseract page-segmentation mode.

### 2. Evaluate on the image-quality test set (PSNR)

```bash
python3 classical/eval_iq.py
```

Reads `best_k` from the tuning summary and runs Wiener across all 16 noise levels of `BMVC_image_quality_test_data`. For each level, averages PSNR of blurred input vs. original (baseline) and Wiener-restored vs. original, then plots both curves.

Outputs:
- `classical/outputs/eval_iq/results.csv`
- `classical/outputs/eval_iq/psnr_vs_noise.png`

Cost: under a minute (no Tesseract). Useful flags:
- `--k 0.01` — override the tuned `k`.
- `--k-summary PATH` — point at a different summary file.
- `--noise-levels n_00,n_05` — restrict the sweep.
- `--limit N` — cap images per level.

### 3. Evaluate on the OCR test set (CER/WER)

```bash
python3 classical/eval_ocr.py
```

Reads `best_k` from the tuning summary and runs Wiener across all 11 noise levels of `BMVC_OCR_test_data`. For each level, averages CER and WER of blurred input vs. original (baseline) and Wiener-restored vs. original, then plots both curves.

Outputs:
- `classical/outputs/eval_ocr/results.csv`
- `classical/outputs/eval_ocr/per_image.csv`
- `classical/outputs/eval_ocr/cer_vs_noise.png`
- `classical/outputs/eval_ocr/wer_vs_noise.png`

Cost: ~35-40 minutes (Tesseract is run on every image at every noise level). Same flags as `eval_iq.py`.

### Run the whole pipeline in one go

```bash
python3 classical/tune_k.py --limit 200 \
  && python3 classical/eval_iq.py \
  && python3 classical/eval_ocr.py
```

Total wall time ~60-65 minutes. After the first run, you can re-run any stage independently.

## Utility scripts

### `ocr/tesseract_eval.py`

Standalone Tesseract evaluator: point it at a blur folder + an orig folder, it pairs files by numeric ID, runs OCR on both, and reports CER/WER. Used under the hood by `eval_ocr.py`, but also runnable directly:

```bash
python3 ocr/tesseract_eval.py \
  --blur-dir data/BMVC_OCR_test_data/n_00 \
  --orig-dir data/BMVC_OCR_test_data/orig \
  --limit 100
```

### `classical/make_preview_train.py`

Produces a 3-up comparison PNG (`Blur | Original | Wiener Restored`) for a single chosen image ID. If the restored copy isn't on disk yet, it computes it on the fly. Good for visually sanity-checking Wiener's behavior.

```bash
python3 classical/make_preview_train.py
```

Result lands in `classical/outputs/previews/`.

### `classical/wienerfiltering.py` (direct CLI)

The underlying Wiener batch driver. Normally called via the scripts above, but you can invoke it directly to produce restored PNGs for a specific folder pair:

```bash
python3 classical/wienerfiltering.py \
  --blur-dir data/BMVC_image_data/blur \
  --psf-dir  data/BMVC_image_data/psf \
  --output-dir classical/outputs/wiener_test \
  --limit 5 --k 0.01
```

## Interpreting the results

Two separate plots come out of the full pipeline:

- **`psnr_vs_noise.png`** — PSNR gain from Wiener tends to be largest at `n_00` and shrinks as noise increases (Wiener amplifies noise at high frequencies).
- **`cer_vs_noise.png`** and **`wer_vs_noise.png`** — the story you actually care about. A "best k for PSNR" is not necessarily the "best k for OCR," because pixel-level metrics reward sharpness even when it introduces ringing artifacts that confuse Tesseract. Tuning on CER (step 1) and reporting on both test sets (steps 2 and 3) lets you see that trade-off.

## Notes for contributors

- `classical/metrics.py` contains the shared helpers so new scripts don't duplicate ID-matching or PSNR logic.
- `tune_k.py`, `eval_iq.py`, and `eval_ocr.py` all follow the same CLI conventions (`--limit`, `--noise-levels` where applicable, `--k-summary` / `--k`, `--output-dir`). Keep that consistent when adding new evaluators.
- New deblurring methods (Richardson-Lucy, learned CNN, etc.) should match the interface `restore(blur_arr, psf_arr, ...) -> restored_arr` so they can drop into the same evaluation scripts with minimal plumbing.
