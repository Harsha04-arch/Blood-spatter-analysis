"""
Image preprocessing pipelines.

Two separate pipelines:
  1. DL preprocessing — for ResNet50 inference / training
  2. CV preprocessing  — for OpenCV droplet detection
"""

import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch
from pathlib import Path
from backend.utils.logger import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
#  DL pipeline                                                                 #
# --------------------------------------------------------------------------- #

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def dl_inference_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def dl_train_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((image_size + 32, image_size + 32)),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def preprocess_for_inference(image_path: Path, image_size: int = 224) -> tuple[torch.Tensor, Image.Image]:
    """
    Returns (tensor, original_pil_image).
    Raises ValueError for invalid / unreadable images.
    """
    img = _load_pil(image_path)
    transform = dl_inference_transform(image_size)
    tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
    return tensor, img


def _load_pil(path: Path) -> Image.Image:
    try:
        img = Image.open(path).convert("RGB")
        img.verify()
    except Exception:
        pass  # verify() closes the file
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot open image {path}: {e}")
    return img


# --------------------------------------------------------------------------- #
#  CV pipeline                                                                 #
# --------------------------------------------------------------------------- #

def preprocess_for_cv(
    image_path: Path,
    threshold: int = 127,
    kernel_size: int = 3,
) -> dict:
    """
    Full OpenCV preprocessing pipeline for droplet extraction.

    Returns dict with keys:
      original, gray, blurred, thresh, morph
    All as NumPy uint8 arrays.
    """
    # Load
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"OpenCV cannot read image: {image_path}")

    original = bgr.copy()

    # Grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Noise reduction — Gaussian blur
    ksize = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    blurred = cv2.GaussianBlur(gray, (ksize * 2 + 1, ksize * 2 + 1), 0)

    # Contrast enhancement — CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # Thresholding — Otsu's method if threshold == 0, else manual
    if threshold == 0:
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(enhanced, threshold, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN,  kernel, iterations=1)

    return {
        "original": original,
        "gray":     gray,
        "blurred":  blurred,
        "enhanced": enhanced,
        "thresh":   thresh,
        "morph":    morph,
    }


def encode_image_base64(img_array: np.ndarray) -> str:
    """Encode a NumPy BGR/gray array to base64 PNG string for JSON transport."""
    import base64
    _, buf = cv2.imencode(".png", img_array)
    return "data:image/png;base64," + base64.b64encode(buf).decode()
