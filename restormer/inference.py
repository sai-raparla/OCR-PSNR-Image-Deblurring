"""Shared inference helpers for Restormer.

Loads a checkpoint and runs inference on numpy arrays / PIL images.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

from model import Restormer, build_model


def pick_device(pref: str = "auto") -> torch.device:
    pref = pref.lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


def load_checkpoint(ckpt_path: str, device: torch.device,
                    variant_override: Optional[str] = None) -> Restormer:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Train one first with `python3 restormer/train.py`."
        )
    state = torch.load(ckpt_path, map_location=device)
    variant = variant_override or state.get("variant", "small")
    model = build_model(variant).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def _np_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """numpy -> float32 tensor robust to numpy/torch ABI mismatches."""
    arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    try:
        return torch.from_numpy(arr).float()
    except RuntimeError:
        t = torch.frombuffer(bytearray(arr.tobytes()), dtype=torch.float32)
        return t.view(*arr.shape)


def _tensor_to_np(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu().contiguous().float()
    try:
        return t.numpy()
    except RuntimeError:
        return np.array(t.tolist(), dtype=np.float32)


@torch.no_grad()
def restore_np(model: Restormer, blur_arr: np.ndarray, device: torch.device) -> np.ndarray:
    """Run Restormer on a single float32 [0, 1] (H, W) numpy array."""
    if blur_arr.ndim != 2:
        raise ValueError(f"expected (H, W) grayscale, got shape {blur_arr.shape}")

    x = _np_to_tensor(blur_arr).unsqueeze(0).unsqueeze(0).to(device)
    y = model(x).clamp(0.0, 1.0).squeeze(0).squeeze(0)
    return _tensor_to_np(y)


@torch.no_grad()
def restore_batch(model: Restormer, blur_arrs: list, device: torch.device) -> list:
    """Batched restoration for equally-shaped inputs."""
    if not blur_arrs:
        return []
    shapes = {a.shape for a in blur_arrs}
    if len(shapes) != 1:
        return [restore_np(model, a, device) for a in blur_arrs]

    stack = np.stack(blur_arrs, axis=0)[:, None, ...].astype(np.float32)
    x = _np_to_tensor(stack).to(device)
    y = model(x).clamp(0.0, 1.0)
    outs = []
    for i in range(y.shape[0]):
        outs.append(_tensor_to_np(y[i, 0]))
    return outs
