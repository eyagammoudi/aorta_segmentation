"""Loading, resampling and augmentation of CTA volumes and aorta masks.

Volumes are read with SimpleITK so that spacing, origin and direction are kept.
Each scan is resampled to a fixed grid for the network; predictions are resampled
back to the original grid for evaluation (see evaluate.py).
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


def list_cases(volumes_dir: str, labels_dir: str) -> list[tuple[str, str]]:
    """Pair volumes and labels by identical file name."""
    vols = sorted(f for f in os.listdir(volumes_dir) if f.endswith((".nii", ".nii.gz")))
    labs = set(os.listdir(labels_dir))
    missing = [f for f in vols if f not in labs]
    if missing:
        raise FileNotFoundError(f"No label found for: {missing[:5]}")
    return [(os.path.join(volumes_dir, f), os.path.join(labels_dir, f)) for f in vols]


def make_splits(cases, val_frac: float, test_frac: float, seed: int, out_path: str | None = None):
    """Patient-level train/val/test split, saved to JSON so it is reproducible."""
    cases = list(cases)
    random.Random(seed).shuffle(cases)
    n = len(cases)
    n_test, n_val = round(n * test_frac), round(n * val_frac)
    splits = {"test": cases[:n_test], "val": cases[n_test:n_test + n_val], "train": cases[n_test + n_val:]}
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(splits, f, indent=2)
    return splits


def resample_to_size(img: sitk.Image, size_xyz, is_label: bool) -> sitk.Image:
    """Resample an image to a fixed voxel grid covering the same physical extent."""
    in_size, in_spacing = np.array(img.GetSize()), np.array(img.GetSpacing())
    out_size = np.array(size_xyz)
    out_spacing = in_spacing * in_size / out_size
    return sitk.Resample(
        img, [int(s) for s in out_size], sitk.Transform(),
        sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear,
        img.GetOrigin(), out_spacing.tolist(), img.GetDirection(), 0.0,
        img.GetPixelID(),
    )


def resample_like(img: sitk.Image, reference: sitk.Image, is_label: bool) -> sitk.Image:
    """Resample img onto the grid of reference (used to map predictions back)."""
    return sitk.Resample(img, reference, sitk.Transform(),
                         sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear, 0.0,
                         img.GetPixelID())


def normalize_hu(arr: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    """Clip to a Hounsfield window and scale to [0, 1] with fixed bounds
    (the same mapping for every scan, unlike per-volume min-max)."""
    arr = np.clip(arr, hu_min, hu_max)
    return ((arr - hu_min) / (hu_max - hu_min)).astype(np.float32)


def load_case(vol_path: str, lab_path: str | None, size_xyz, hu_window):
    """Return (image array, label array or None, original SimpleITK image).
    Arrays are in (D, H, W) order as returned by SimpleITK."""
    vol = sitk.ReadImage(vol_path, sitk.sitkFloat32)
    vol_r = resample_to_size(vol, size_xyz, is_label=False)
    x = normalize_hu(sitk.GetArrayFromImage(vol_r), *hu_window)
    y = None
    if lab_path is not None:
        lab = sitk.ReadImage(lab_path, sitk.sitkUInt8)
        if lab.GetSize() == vol.GetSize():
            lab.CopyInformation(vol)  # guard against tiny header mismatches
        lab_r = resample_to_size(lab, size_xyz, is_label=True)
        # All aortic labels (aorta + branches) are merged into one foreground class.
        y = (sitk.GetArrayFromImage(lab_r) > 0).astype(np.float32)
    return x, y, vol


def augment(x: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """Light 3D augmentation: random flips and intensity scale/shift."""
    for axis in range(3):
        if rng.random() < 0.5:
            x, y = np.flip(x, axis), np.flip(y, axis)
    x = x * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05)
    return np.ascontiguousarray(x, dtype=np.float32), np.ascontiguousarray(y, dtype=np.float32)


class AortaDataset(Dataset):
    def __init__(self, cases, size_xyz, hu_window, train: bool = False, seed: int = 0):
        self.cases, self.size_xyz, self.hu_window, self.train = cases, size_xyz, hu_window, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        x, y, _ = load_case(*self.cases[i], self.size_xyz, self.hu_window)
        if self.train:
            x, y = augment(x, y, self.rng)
        return torch.from_numpy(x)[None], torch.from_numpy(y)[None]
