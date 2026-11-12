"""Overlap and boundary metrics on binary masks (numpy, D x H x W)."""
import numpy as np
from scipy import ndimage


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else 2.0 * np.logical_and(pred, gt).sum() / denom


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else np.logical_and(pred, gt).sum() / union


def _surface(mask):
    return mask & ~ndimage.binary_erosion(mask)


def hd95(pred: np.ndarray, gt: np.ndarray, spacing_dhw) -> float:
    """95th-percentile symmetric Hausdorff distance in mm."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() or not gt.any():
        return float("nan")
    sp, sg = _surface(pred), _surface(gt)
    dt_g = ndimage.distance_transform_edt(~sg, sampling=spacing_dhw)
    dt_p = ndimage.distance_transform_edt(~sp, sampling=spacing_dhw)
    return float(np.percentile(np.concatenate([dt_g[sp], dt_p[sg]]), 95))
