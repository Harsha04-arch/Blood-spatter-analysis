"""
Inference module — runs the BPA classifier on a single image.

Returns structured results with spatter type, velocity category,
confidence scores, and top-K predictions.

IMPORTANT: Results are labeled as AI-based classification/estimation,
not direct physical measurements.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from backend.ml.model import load_model, get_device, _SPATTER_CLASSES, _VELOCITY_CLASSES
from backend.ml.preprocessing import preprocess_for_inference
from backend.utils.logger import get_logger

log = get_logger(__name__)


def predict(image_path: Path, model_path: Path, image_size: int = 224, top_k: int = 3) -> dict:
    """
    Run inference on a single image.

    Returns:
        dict with spatter prediction, velocity prediction, probabilities, top-k results.

    Raises:
        FileNotFoundError — model checkpoint missing
        ValueError        — invalid image
    """
    device = get_device()
    model = load_model(model_path, device)

    tensor, pil_img = preprocess_for_inference(image_path, image_size)
    tensor = tensor.to(device)

    with torch.no_grad():
        spatter_logits, velocity_logits = model(tensor)

    spatter_probs  = F.softmax(spatter_logits,  dim=1).cpu().squeeze()
    velocity_probs = F.softmax(velocity_logits, dim=1).cpu().squeeze()

    # Spatter results
    spatter_conf, spatter_idx = spatter_probs.max(0)
    spatter_type = _SPATTER_CLASSES[spatter_idx.item()]
    spatter_confidence = round(spatter_conf.item() * 100, 2)

    # Top-K spatter
    top_k = min(top_k, len(_SPATTER_CLASSES))
    top_vals, top_idxs = spatter_probs.topk(top_k)
    top_predictions = [
        {
            "rank": i + 1,
            "class": _SPATTER_CLASSES[idx.item()],
            "probability": round(val.item() * 100, 2),
        }
        for i, (val, idx) in enumerate(zip(top_vals, top_idxs))
    ]

    # All class probabilities
    all_spatter_probs = {
        cls: round(prob.item() * 100, 2)
        for cls, prob in zip(_SPATTER_CLASSES, spatter_probs)
    }

    # Velocity results
    vel_conf, vel_idx = velocity_probs.max(0)
    velocity_class = _VELOCITY_CLASSES[vel_idx.item()]
    velocity_confidence = round(vel_conf.item() * 100, 2)

    all_velocity_probs = {
        cls: round(prob.item() * 100, 2)
        for cls, prob in zip(_VELOCITY_CLASSES, velocity_probs)
    }

    return {
        "disclaimer": (
            "AI-based classification/estimation — "
            "not a direct physical measurement. "
            "Results require validation by qualified forensic professionals."
        ),
        "spatter": {
            "predicted_type":  spatter_type,
            "confidence":      spatter_confidence,
            "top_predictions": top_predictions,
            "all_probabilities": all_spatter_probs,
            "label": "AI Prediction",
        },
        "velocity": {
            "category":   velocity_class,
            "confidence": velocity_confidence,
            "all_probabilities": all_velocity_probs,
            "label": "Model-based Classification",
        },
        "image_size": list(pil_img.size),
    }
