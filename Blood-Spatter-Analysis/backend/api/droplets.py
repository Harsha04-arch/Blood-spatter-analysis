"""Droplets API — Module 3 endpoints."""

import base64
import cv2
import numpy as np
from pathlib import Path
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from backend.config import Config
from backend.ml.preprocessing import preprocess_for_cv, encode_image_base64
from backend.computer_vision.droplet_detection import detect_droplets
from backend.database.models import AnalysisModel, get_db
from backend.utils.logger import get_logger

log = get_logger(__name__)
droplets_bp = Blueprint("droplets", __name__, url_prefix="/api/droplets")


def _get_model():
    db = get_db(Config.DB_PATH)
    return AnalysisModel(db)


@droplets_bp.route("/extract", methods=["POST"])
def extract_droplets():
    """
    Extract droplets from an uploaded or existing image.
    Returns droplet measurements + base64-encoded annotated image.
    """
    case_id = request.form.get("case_id", "general")
    image_name = request.form.get("image_name", "")
    min_area = int(request.form.get("min_area", Config.DEFAULT_MIN_DROPLET_AREA))
    max_area = int(request.form.get("max_area", Config.DEFAULT_MAX_DROPLET_AREA))
    threshold = int(request.form.get("threshold", Config.DEFAULT_THRESHOLD))
    kernel_size = int(request.form.get("kernel_size", Config.DEFAULT_KERNEL_SIZE))

    # Resolve image path
    if "file" in request.files:
        f = request.files["file"]
        fname = secure_filename(f.filename)
        save_path = Config.UPLOAD_FOLDER / case_id / fname
        save_path.parent.mkdir(parents=True, exist_ok=True)
        f.save(str(save_path))
        image_name = fname
    elif image_name:
        save_path = Config.UPLOAD_FOLDER / case_id / secure_filename(image_name)
        if not save_path.exists():
            return jsonify({"error": f"Image not found: {image_name}"}), 404
    else:
        return jsonify({"error": "Provide a file or image_name."}), 400

    try:
        stages = preprocess_for_cv(save_path, threshold=threshold, kernel_size=kernel_size)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    droplets, annotated = detect_droplets(
        morph=stages["morph"],
        original=stages["original"],
        min_area=min_area,
        max_area=max_area,
    )

    if not droplets:
        return jsonify({
            "status": "no_droplets",
            "message": (
                "No droplets detected. Try adjusting threshold, "
                "min_area, or kernel_size parameters."
            ),
            "droplets": [],
            "count": 0,
        }), 200

    droplet_dicts = [d.to_dict() for d in droplets]

    # Build summary stats
    valid_angles = [d["impact_angle"] for d in droplet_dicts if d["angle_valid"]]
    summary = {
        "count": len(droplets),
        "valid_angles": len(valid_angles),
        "avg_angle":   round(float(np.mean(valid_angles)), 2) if valid_angles else None,
        "min_angle":   round(float(np.min(valid_angles)), 2)  if valid_angles else None,
        "max_angle":   round(float(np.max(valid_angles)), 2)  if valid_angles else None,
        "avg_area":    round(float(np.mean([d["area"] for d in droplet_dicts])), 2),
        "disclaimer":  "Computer Vision Measurement — accuracy depends on image quality and calibration.",
    }

    # Encode annotated image
    annotated_b64 = encode_image_base64(annotated)
    thresh_b64 = encode_image_base64(stages["thresh"])
    morph_b64  = encode_image_base64(stages["morph"])

    result = {
        "status": "success",
        "image_name": image_name,
        "case_id": case_id,
        "droplets": droplet_dicts,
        "summary": summary,
        "annotated_image": annotated_b64,
        "threshold_image": thresh_b64,
        "morphology_image": morph_b64,
    }

    # Persist
    if case_id != "general":
        _get_model().save(case_id, image_name, {"droplets": {"droplets": droplet_dicts, "summary": summary}})

    return jsonify(result)


@droplets_bp.route("/angle", methods=["POST"])
def compute_angle():
    """Compute impact angle from raw W and L values."""
    data = request.get_json(silent=True) or {}
    try:
        w = float(data["width"])
        l = float(data["length"])
    except (KeyError, ValueError):
        return jsonify({"error": "Provide 'width' (W) and 'length' (L) as numbers."}), 400

    if l <= 0:
        return jsonify({"error": "Length L must be positive."}), 400

    ratio = w / l
    if not (0.0 <= ratio <= 1.0):
        return jsonify({
            "error": f"W/L = {ratio:.4f} is outside valid range [0, 1].",
            "note": "Check measurements for segmentation errors.",
        }), 400

    import math
    angle = math.degrees(math.asin(ratio))
    return jsonify({
        "W": w, "L": l,
        "ratio": round(ratio, 4),
        "impact_angle_degrees": round(angle, 2),
        "formula": "θ = arcsin(W / L)",
        "label": "Geometric Estimate",
    })
