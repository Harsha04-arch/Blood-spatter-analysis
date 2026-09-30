"""
Module 1 — Automated Dataset Fetcher & Ingestion.

Handles:
  - ZIP upload extraction
  - Local folder scanning
  - Image validation (format, dimensions, corruption, duplicates)
  - Auto-organization into class subdirectories
  - Train/val/test splitting (80/10/10, reproducible)
  - Dataset statistics
"""

import hashlib
import json
import random
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from backend.utils.logger import get_logger
from backend.utils.file_utils import file_hash

log = get_logger(__name__)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

DEFAULT_CLASSES = [
    "Passive Drip",
    "Transfer Smear",
    "Medium Velocity Impact Spatter",
    "High Velocity Impact Spatter",
]

SPLITS = {"train": 0.80, "val": 0.10, "test": 0.10}


def ingest_zip(
    zip_path: Path,
    dataset_root: Path,
    dataset_name: str,
    seed: int = 42,
) -> dict:
    """Extract ZIP and ingest."""
    extract_dir = dataset_root / f"_tmp_{dataset_name}"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    result = ingest_folder(extract_dir, dataset_root, dataset_name, seed=seed)
    shutil.rmtree(extract_dir, ignore_errors=True)
    return result


def ingest_folder(
    source_dir: Path,
    dataset_root: Path,
    dataset_name: str,
    classes: Optional[list[str]] = None,
    seed: int = 42,
) -> dict:
    """
    Scan source_dir for images, organise into class directories,
    split into train/val/test.

    source_dir layout accepted:
      a) source_dir/ClassName/image.jpg   (already organised)
      b) source_dir/image.jpg             (flat — assigned to first class)
    """
    dest = dataset_root / dataset_name
    if dest.exists():
        return {
            "status": "error",
            "message": f"Dataset '{dataset_name}' already exists. Delete it first.",
        }

    classes = classes or DEFAULT_CLASSES
    random.seed(seed)

    # ---- Discover images ----
    all_images = _find_images(source_dir)
    if not all_images:
        return {"status": "error", "message": "No valid images found in the source."}

    # ---- Validate & deduplicate ----
    valid, warnings = _validate_images(all_images)

    # ---- Determine class assignments ----
    assignments = _assign_classes(valid, source_dir, classes)

    # ---- Split ----
    splits = _split(assignments, seed)

    # ---- Copy to dest ----
    stats = _copy_and_save(splits, dest)
    stats["dataset_name"] = dataset_name
    stats["classes"] = classes
    stats["warnings"] = warnings
    stats["seed"] = seed
    stats["status"] = "success"

    # Save metadata
    meta_path = dest / "metadata.json"
    meta_path.write_text(json.dumps(stats, indent=2))
    log.info(f"Dataset '{dataset_name}' ingested: {stats['total']} images → {dest}")

    return stats


def get_dataset_stats(dataset_path: Path) -> dict:
    """Load stats from a previously ingested dataset."""
    meta = dataset_path / "metadata.json"
    if not meta.exists():
        return {"status": "error", "message": "Dataset metadata not found."}
    return json.loads(meta.read_text())


# --------------------------------------------------------------------------- #
#  Internal helpers                                                            #
# --------------------------------------------------------------------------- #

def _find_images(root: Path) -> list[Path]:
    found = []
    for ext in ALLOWED_EXTENSIONS:
        found.extend(root.rglob(f"*{ext}"))
        found.extend(root.rglob(f"*{ext.upper()}"))
    return sorted(set(found))


def _validate_images(paths: list[Path]) -> tuple[list[dict], list[str]]:
    valid: list[dict] = []
    seen_hashes: set[str] = set()
    warnings: list[str] = []

    for p in paths:
        try:
            img = Image.open(p)
            img.verify()
        except Exception as e:
            warnings.append(f"Corrupted/unreadable: {p.name} — {e}")
            continue

        try:
            img = Image.open(p)
            w, h = img.size
        except Exception:
            warnings.append(f"Cannot read dimensions: {p.name}")
            continue

        if w < 16 or h < 16:
            warnings.append(f"Too small ({w}×{h}): {p.name}")
            continue

        h_val = file_hash(p)
        if h_val in seen_hashes:
            warnings.append(f"Duplicate skipped: {p.name}")
            continue
        seen_hashes.add(h_val)

        valid.append({"path": p, "width": w, "height": h, "hash": h_val})

    return valid, warnings


def _assign_classes(images: list[dict], source_dir: Path, classes: list[str]) -> dict[str, list[dict]]:
    """Map images to classes based on their parent directory name."""
    assignments: dict[str, list[dict]] = {c: [] for c in classes}
    unclassified = []

    for img in images:
        parent = img["path"].parent.name
        matched = None
        for cls in classes:
            if cls.lower().replace(" ", "_") == parent.lower().replace(" ", "_"):
                matched = cls
                break
            if cls.lower() in parent.lower():
                matched = cls
                break
        if matched:
            assignments[matched].append(img)
        else:
            unclassified.append(img)

    # Distribute unclassified evenly
    if unclassified:
        for i, img in enumerate(unclassified):
            cls = classes[i % len(classes)]
            assignments[cls].append(img)

    return assignments


def _split(assignments: dict, seed: int) -> dict:
    """Split each class into train/val/test."""
    result: dict = {"train": {}, "val": {}, "test": {}}
    random.seed(seed)
    for cls, images in assignments.items():
        random.shuffle(images)
        n = len(images)
        n_val  = max(0, int(n * SPLITS["val"]))
        n_test = max(0, int(n * SPLITS["test"]))
        n_train = n - n_val - n_test
        result["train"][cls] = images[:n_train]
        result["val"][cls]   = images[n_train:n_train + n_val]
        result["test"][cls]  = images[n_train + n_val:]
    return result


def _copy_and_save(splits: dict, dest: Path) -> dict:
    stats: dict = {"splits": {}, "class_distribution": {}, "total": 0}
    for split_name, class_map in splits.items():
        split_total = 0
        for cls, images in class_map.items():
            cls_dir = dest / split_name / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            for img in images:
                shutil.copy2(img["path"], cls_dir / img["path"].name)
            split_total += len(images)
            stats["class_distribution"].setdefault(cls, 0)
            stats["class_distribution"][cls] += len(images)
        stats["splits"][split_name] = split_total
        stats["total"] += split_total
    return stats
