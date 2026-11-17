# 3D Aorta Segmentation from CT Angiography

A 3D U-Net that segments the aorta and its branches in contrast-enhanced CT angiography (CTA) volumes, with an evaluation done in each scan's **native image space** (Dice, IoU and 95th-percentile Hausdorff distance in mm).



## Why it matters

Accurate 3D segmentation of the aorta is the first step for measuring diameters, planning endovascular repair and following aneurysms over time. Vessels are thin, tubular and a small fraction of the volume, so the task is strongly class-imbalanced.

## Method

| Step | Choice |
|---|---|
| Data | CIS-UNet aortic CTA dataset ([Imran et al., 2024](https://arxiv.org/abs/2401.13049)); the aorta and its branch labels are merged into one foreground class |
| Split | Patient-level train / validation / test (70 / 15 / 15), fixed seed, saved to `splits.json` |
| Pre-processing | Read with SimpleITK (spacing and orientation preserved) · resampled to a 128 × 128 × 64 grid · HU window [−200, 800] scaled to [0, 1] |
| Augmentation | Random flips on all three axes, random intensity scale and shift |
| Model | 3D U-Net, 4 resolution levels, 16 → 256 channels, batch norm |
| Loss | 0.5 × soft Dice + 0.5 × binary cross-entropy (from logits) |
| Training | Adam, lr 1e-4, LR reduced on validation-Dice plateau, best model kept by validation Dice |
| Evaluation | Prediction mapped back to the original voxel grid; Dice, IoU, HD95 (mm) on the held-out test set |


## Repository structure

```
aortaseg/
  data.py       loading, resampling, HU normalisation, augmentation, splits
  model.py      3D U-Net
  losses.py     Dice + BCE loss
  metrics.py    Dice, IoU, HD95
  train.py      training loop
  evaluate.py   test-set evaluation in native image space
configs/default.yaml   all settings in one place
notebooks/aorta_segmentation_colab.ipynb   run everything on Google Colab
```

## How to run

```bash
pip install -r requirements.txt
# put the data in data/Volumes and data/Labels (matching file names), or edit configs/default.yaml
python -m aortaseg.train    --config configs/default.yaml
python -m aortaseg.evaluate --config configs/default.yaml --save-predictions
```

Outputs (model, splits, per-case metrics, summary) are written to `outputs/`. No data is included in this repository; obtain the dataset from its authors and follow its terms of use.

## Limitations and next steps

- **Whole-volume resampling** to 128 × 128 × 64 loses detail on small branches; patch-based training at native resolution (e.g. nnU-Net) would be the stronger baseline.
- **Binary task**: the dataset provides the aorta and 13 branches separately; a multi-class version is a natural extension.
- **Topology**: Dice does not penalise broken vessels; a topology-aware loss (e.g. clDice) and connectivity checks would suit tubular structures better.
- **Uncertainty**: Monte Carlo dropout or ensembles could flag unreliable regions for review.

## Author

Eya Gammoudi · Erasmus Mundus EMMBIOME M.Sc., Biomedical Engineering
