"""
Module 2 — Training pipeline for the BPA ResNet50 classifier.

Supports:
  - Configurable epochs, batch size, learning rate, image size, seed
  - Training / validation loops with live progress via callback
  - Test-set evaluation with accuracy, precision, recall, F1, confusion matrix
  - Checkpoint saving
"""

import json
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from backend.ml.model import BPAClassifier, clear_model_cache, _SPATTER_CLASSES, _VELOCITY_CLASSES
from backend.ml.preprocessing import dl_train_transform, dl_inference_transform
from backend.utils.logger import get_logger

log = get_logger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
#  Training entry point                                                        #
# --------------------------------------------------------------------------- #

def train_model(
    dataset_path: Path,
    model_save_path: Path,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    image_size: int = 224,
    seed: int = 42,
    val_split: float = 0.1,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """
    Train the BPA classifier on the provided dataset.

    dataset_path must contain subdirectories per class (ImageFolder format).

    Returns a dict with training history and test metrics.
    Raises ValueError if dataset is insufficient.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training on device: {device}")

    # ---- Dataset ----
    full_dataset = ImageFolder(str(dataset_path), transform=dl_train_transform(image_size))
    if len(full_dataset) < 10:
        raise ValueError(f"Dataset too small: {len(full_dataset)} images. Need at least 10.")

    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    # Use inference transform for validation
    val_ds.dataset.transform = dl_inference_transform(image_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    classes = full_dataset.classes
    log.info(f"Classes: {classes}, Train: {n_train}, Val: {n_val}")

    # ---- Model ----
    n_spatter  = min(len(classes), len(_SPATTER_CLASSES))
    n_velocity = len(_VELOCITY_CLASSES)
    model = BPAClassifier(num_spatter=n_spatter, num_velocity=n_velocity, pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "epochs": epochs, "classes": classes,
    }

    best_val_loss = float("inf")
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # -- Train --
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            spatter_out, velocity_out = model(imgs)
            # Use spatter labels for primary head; velocity head shares label space
            loss_s = criterion(spatter_out, labels)
            # Velocity labels: map spatter label → velocity bucket
            vel_labels = _spatter_to_velocity(labels, n_velocity)
            loss_v = criterion(velocity_out, vel_labels)
            loss = loss_s + 0.3 * loss_v
            loss.backward()
            optimizer.step()
            t_loss   += loss_s.item() * imgs.size(0)
            preds     = spatter_out.argmax(1)
            t_correct += (preds == labels).sum().item()
            t_total   += imgs.size(0)
        scheduler.step()
        train_loss = t_loss / t_total
        train_acc  = t_correct / t_total * 100

        # -- Validate --
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                spatter_out, _ = model(imgs)
                loss = criterion(spatter_out, labels)
                v_loss   += loss.item() * imgs.size(0)
                preds     = spatter_out.argmax(1)
                v_correct += (preds == labels).sum().item()
                v_total   += imgs.size(0)
        val_loss = v_loss / v_total
        val_acc  = v_correct / v_total * 100

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_acc"].append(round(train_acc, 2))
        history["val_acc"].append(round(val_acc, 2))

        log.info(f"Epoch {epoch}/{epochs} — train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                 f"train_acc={train_acc:.1f}% val_acc={val_acc:.1f}%")

        if progress_callback:
            progress_callback({
                "epoch": epoch, "total_epochs": epochs,
                "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4),
                "train_acc": round(train_acc, 2), "val_acc": round(val_acc, 2),
            })

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "classes": classes,
                "n_spatter": n_spatter,
                "n_velocity": n_velocity,
            }, str(model_save_path))
            log.info(f"  ✓ Best model saved (val_loss={val_loss:.4f})")

    # ---- Test metrics on val set ----
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            spatter_out, _ = model(imgs)
            preds = spatter_out.argmax(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    metrics = {
        "accuracy":  round(accuracy_score(all_labels, all_preds) * 100, 2),
        "precision": round(precision_score(all_labels, all_preds, average="weighted", zero_division=0) * 100, 2),
        "recall":    round(recall_score(all_labels, all_preds, average="weighted", zero_division=0) * 100, 2),
        "f1_score":  round(f1_score(all_labels, all_preds, average="weighted", zero_division=0) * 100, 2),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "classes": classes,
    }

    clear_model_cache()  # force reload from saved checkpoint next inference

    return {
        "status": "completed",
        "history": history,
        "metrics": metrics,
        "model_path": str(model_save_path),
        "best_val_loss": round(best_val_loss, 4),
    }


def _spatter_to_velocity(labels: torch.Tensor, n_velocity: int) -> torch.Tensor:
    """Map spatter class index to velocity class index heuristically."""
    # 0: Passive Drip → 0 Low
    # 1: Transfer Smear → 0 Low
    # 2: MVIS → 1 Medium
    # 3: HVIS → 2 High
    mapping = {0: 0, 1: 0, 2: 1, 3: 2}
    vel = labels.clone()
    for i, lbl in enumerate(labels):
        vel[i] = mapping.get(lbl.item(), lbl.item() % n_velocity)
    return vel
