"""Train the 3D U-Net.  Usage:  python -m aortaseg.train --config configs/default.yaml"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from .data import AortaDataset, list_cases, make_splits
from .losses import DiceBCELoss
from .model import UNet3D
from .utils import get_device, load_config, seed_everything


@torch.no_grad()
def soft_dice(logits, y):
    p = (torch.sigmoid(logits) > 0.5).float()
    return ((2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)).item()


def main(cfg):
    seed_everything(cfg["seed"])
    device = get_device()
    out = cfg["output_dir"]
    os.makedirs(out, exist_ok=True)

    cases = list_cases(cfg["data"]["volumes_dir"], cfg["data"]["labels_dir"])
    splits = make_splits(cases, cfg["data"]["val_frac"], cfg["data"]["test_frac"], cfg["seed"],
                         os.path.join(out, "splits.json"))
    print({k: len(v) for k, v in splits.items()})

    size, hu = tuple(cfg["data"]["size_xyz"]), tuple(cfg["data"]["hu_window"])
    train_dl = DataLoader(AortaDataset(splits["train"], size, hu, train=True, seed=cfg["seed"]),
                          batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"])
    val_dl = DataLoader(AortaDataset(splits["val"], size, hu), batch_size=1, num_workers=cfg["train"]["num_workers"])

    model = UNet3D(features=cfg["model"]["features"]).to(device)
    loss_fn = DiceBCELoss(cfg["train"]["dice_weight"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=5, factor=0.5)

    history, best = [], -1.0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        tr_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() / len(train_dl)

        model.eval()
        with torch.no_grad():
            val_dice = sum(soft_dice(model(x.to(device)), y.to(device)) for x, y in val_dl) / max(len(val_dl), 1)
        sched.step(val_dice)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_dice": val_dice})
        print(f"epoch {epoch:3d}  train_loss {tr_loss:.4f}  val_dice {val_dice:.4f}")
        if val_dice > best:
            best = val_dice
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": val_dice, "config": cfg},
                       os.path.join(out, "best_model.pt"))

    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Best validation Dice: {best:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    main(load_config(ap.parse_args().config))
