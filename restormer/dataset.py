"""Paired blur/orig datasets for training and evaluating Restormer.

`BMVCPairedDataset` loads the training pool `data/BMVC_image_data/` which has a
single blur level; it performs random cropping and flips to produce fixed-size
patches for SGD.

`PairedDirDataset` loads a specific (blur_dir, orig_dir) combo with no
augmentation, used for evaluation (the n_XX folders of the IQ / OCR test sets).
"""

from __future__ import annotations

import os
import random
import re
from typing import Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_NUM_RE = re.compile(r"(\d+)")


def _id_map(folder: str) -> dict:
    d = {}
    for name in os.listdir(folder):
        if not name.lower().endswith(".png"):
            continue
        m = _NUM_RE.match(name)
        if m is None:
            continue
        d[m.group(1)] = os.path.join(folder, name)
    return d


def load_gray(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    """NumPy -> float32 torch tensor, robust to numpy/torch ABI mismatches.

    Prefers `torch.from_numpy` (zero-copy). If the user has a torch wheel
    compiled against a different NumPy major version (common on macOS with
    numpy>=2 + older torch), that path raises `RuntimeError: Numpy is not
    available`; we fall back to a bytes-level copy via `torch.frombuffer`.
    """
    if arr.ndim == 2:
        arr = arr[None, ...]
    arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    try:
        return torch.from_numpy(arr).float()
    except RuntimeError:
        t = torch.frombuffer(bytearray(arr.tobytes()), dtype=torch.float32)
        return t.view(*arr.shape)


# ---------------------------------------------------------------------------
# Training set: random crops with flips
# ---------------------------------------------------------------------------


class BMVCPairedDataset(Dataset):
    """Random-crop paired dataset from `BMVC_image_data/{blur,orig}`.

    Every __getitem__ returns a `(blur_patch, orig_patch)` tensor pair of
    shape (1, patch, patch) with values in [0, 1].
    """

    def __init__(
        self,
        blur_dir: str,
        orig_dir: str,
        patch_size: int = 128,
        augment: bool = True,
        limit: Optional[int] = None,
        seed: int = 0,
    ):
        self.blur_dir = blur_dir
        self.orig_dir = orig_dir
        self.patch_size = patch_size
        self.augment = augment

        blur_m = _id_map(blur_dir)
        orig_m = _id_map(orig_dir)
        ids = sorted(set(blur_m) & set(orig_m))

        if limit is not None:
            ids = ids[:limit]

        self.pairs = [(blur_m[i], orig_m[i]) for i in ids]
        if not self.pairs:
            raise RuntimeError(
                f"No paired images found in {blur_dir} / {orig_dir}"
            )

        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def _random_crop_pair(self, blur: np.ndarray, orig: np.ndarray):
        if blur.shape != orig.shape:
            raise ValueError(
                f"shape mismatch blur={blur.shape} orig={orig.shape}"
            )
        h, w = blur.shape
        ps = self.patch_size

        if h < ps or w < ps:
            pad_h = max(0, ps - h)
            pad_w = max(0, ps - w)
            blur = np.pad(blur, ((0, pad_h), (0, pad_w)), mode="reflect")
            orig = np.pad(orig, ((0, pad_h), (0, pad_w)), mode="reflect")
            h, w = blur.shape

        top = self._rng.randint(0, h - ps)
        left = self._rng.randint(0, w - ps)
        blur = blur[top:top + ps, left:left + ps]
        orig = orig[top:top + ps, left:left + ps]
        return blur, orig

    def _augment_pair(self, blur: np.ndarray, orig: np.ndarray):
        if self._rng.random() < 0.5:
            blur = np.fliplr(blur).copy()
            orig = np.fliplr(orig).copy()
        if self._rng.random() < 0.5:
            blur = np.flipud(blur).copy()
            orig = np.flipud(orig).copy()
        k = self._rng.randint(0, 3)
        if k:
            blur = np.rot90(blur, k).copy()
            orig = np.rot90(orig, k).copy()
        return blur, orig

    def __getitem__(self, idx: int):
        blur_path, orig_path = self.pairs[idx]
        blur = load_gray(blur_path)
        orig = load_gray(orig_path)

        blur, orig = self._random_crop_pair(blur, orig)
        if self.augment:
            blur, orig = self._augment_pair(blur, orig)

        return to_tensor(blur), to_tensor(orig)


# ---------------------------------------------------------------------------
# Eval set: whole images, no augmentation
# ---------------------------------------------------------------------------


class PairedDirDataset(Dataset):
    """Full-image paired loader for validation / evaluation."""

    def __init__(self, blur_dir: str, orig_dir: str, limit: Optional[int] = None):
        blur_m = _id_map(blur_dir)
        orig_m = _id_map(orig_dir)
        ids = sorted(set(blur_m) & set(orig_m))
        if limit is not None:
            ids = ids[:limit]

        self.ids = ids
        self.pairs = [(blur_m[i], orig_m[i]) for i in ids]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        blur_path, orig_path = self.pairs[idx]
        blur = load_gray(blur_path)
        orig = load_gray(orig_path)
        return self.ids[idx], to_tensor(blur), to_tensor(orig)
