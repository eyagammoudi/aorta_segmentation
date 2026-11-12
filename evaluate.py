"""Evaluate the best model on the held-out test set, in the ORIGINAL image space.

Predictions are made on the resampled grid, mapped back to each scan's native grid,
and compared with the original labels (Dice, IoU, HD95 in mm).

Usage:  python -m aortaseg.evaluate --config configs/default.yaml [--save-predictions]
"""
import argparse
import csv
import json
import os

import numpy as np
import SimpleITK as sitk
import torch

from .data import load_case, resample_like
from .metrics import dice, hd95, iou
from .model import UNet3D
from .utils import get_device, load_config


@torch.no_grad()
def predict(model, vol_path, size_xyz, hu_window, device):
    x, _, ref = load_case(vol_path, None, size_xyz, hu_window)
    prob = torch.sigmoid(model(torch.from_numpy(x)[None, None].to(device)))[0, 0].cpu().numpy()
    small = sitk.GetImageFromArray((prob > 0.5).astype(np.uint8))
    # Recreate the resampled grid geometry, then map back to the native grid.
    spacing = np.array(ref.GetSpacing()) * np.array(ref.GetSize()) / np.array(size_xyz)
    small.SetSpacing(spacing.tolist()); small.SetOrigin(ref.GetOrigin()); small.SetDirection(ref.GetDirection())
    return resample_like(small, ref, is_label=True), ref


def main(cfg, save_predictions: bool):
    device, out = get_device(), cfg["output_dir"]
    ckpt = torch.load(os.path.join(out, "best_model.pt"), map_location=device, weights_only=False)
    model = UNet3D(features=cfg["model"]["features"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    with open(os.path.join(out, "splits.json")) as f:
        test = json.load(f)["test"]

    size, hu = tuple(cfg["data"]["size_xyz"]), tuple(cfg["data"]["hu_window"])
    rows = []
    for vol_path, lab_path in test:
        pred_img, ref = predict(model, vol_path, size, hu, device)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(lab_path)) > 0
        pred = sitk.GetArrayFromImage(pred_img) > 0
        spacing_dhw = ref.GetSpacing()[::-1]
        rows.append({"case": os.path.basename(vol_path), "dice": dice(pred, gt),
                     "iou": iou(pred, gt), "hd95_mm": hd95(pred, gt, spacing_dhw)})
        print(rows[-1])
        if save_predictions:
            os.makedirs(os.path.join(out, "predictions"), exist_ok=True)
            sitk.WriteImage(pred_img, os.path.join(out, "predictions", os.path.basename(vol_path)))

    with open(os.path.join(out, "test_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    summary = {k: {"mean": float(np.nanmean([r[k] for r in rows])), "std": float(np.nanstd([r[k] for r in rows]))}
               for k in ("dice", "iou", "hd95_mm")}
    summary["n_test_cases"] = len(rows)
    with open(os.path.join(out, "test_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--save-predictions", action="store_true")
    a = ap.parse_args()
    main(load_config(a.config), a.save_predictions)
