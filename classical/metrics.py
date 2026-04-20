"""Shared helpers for the classical Wiener pipeline scripts.

Provides:
    - psnr(ref, restored): Peak Signal-to-Noise Ratio for float32 arrays in [0, 1].
    - match_ids_3way(blur_dir, psf_dir, orig_dir): intersect IDs across three folders.
    - wiener_restore_arr(blur_arr, psf_arr, k): thin wrapper around do_wiener.
"""

import os
import re
import numpy as np

from wienerfiltering import do_wiener


def psnr(ref: np.ndarray, restored: np.ndarray) -> float:
    ref = np.asarray(ref, dtype=np.float32)
    restored = np.asarray(restored, dtype=np.float32)

    if ref.shape != restored.shape:
        raise ValueError(
            f"PSNR shape mismatch: ref {ref.shape} vs restored {restored.shape}"
        )

    diff = ref - restored
    mse = float(np.mean(diff * diff))

    if mse <= 1e-12:
        return float("inf")

    return 10.0 * float(np.log10(1.0 / mse))


def _id_to_filename_map(folder: str) -> dict:
    d = {}
    for name in os.listdir(folder):
        if not name.lower().endswith(".png"):
            continue
        m = re.match(r"(\d+)", name)
        if m is None:
            continue
        d[m.group(1)] = name
    return d


def match_ids_3way(blur_dir: str, psf_dir: str, orig_dir: str):
    """Return (sorted_ids, blur_map, psf_map, orig_map).

    sorted_ids is the sorted intersection of numeric IDs present in all three folders.
    Each *_map is a {id: filename} dict for lookups.
    """
    blur_map = _id_to_filename_map(blur_dir)
    psf_map = _id_to_filename_map(psf_dir)
    orig_map = _id_to_filename_map(orig_dir)

    common = set(blur_map) & set(psf_map) & set(orig_map)
    sorted_ids = sorted(common)

    return sorted_ids, blur_map, psf_map, orig_map


def wiener_restore_arr(blur_arr: np.ndarray, psf_arr: np.ndarray, k: float) -> np.ndarray:
    """Deconvolve a single blurred image with Wiener using the given PSF and k."""
    return do_wiener(blur_arr, psf_arr, k)
