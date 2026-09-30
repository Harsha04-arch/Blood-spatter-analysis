"""Convergence & 3D Point of Origin API — Module 4 endpoints."""

from flask import Blueprint, request, jsonify

from backend.geometry.convergence_2d import compute_2d_convergence
from backend.geometry.origin_3d import compute_3d_origin
from backend.database.models import AnalysisModel, get_db
from backend.config import Config
from backend.utils.logger import get_logger

log = get_logger(__name__)
convergence_bp = Blueprint("convergence", __name__, url_prefix="/api")


def _get_analysis_model():
    db = get_db(Config.DB_PATH)
    return AnalysisModel(db)


@convergence_bp.route("/convergence/2d", methods=["POST"])
def convergence_2d():
    """
    Compute 2D area of convergence from a list of droplets.

    Body JSON:
      { "droplets": [...], "case_id": "...", "image_name": "..." }
    """
    data = request.get_json(silent=True) or {}
    droplets = data.get("droplets", [])
    case_id  = data.get("case_id", "")
    img_name = data.get("image_name", "")

    if not droplets:
        return jsonify({"error": "No droplet data provided."}), 400

    result = compute_2d_convergence(droplets)

    if case_id:
        _get_analysis_model().save(case_id, img_name, {"convergence_2d": result})

    return jsonify(result)


@convergence_bp.route("/reconstruction/3d", methods=["POST"])
def reconstruction_3d():
    """
    Compute 3D point of origin.

    Body JSON:
      {
        "droplets": [...],
        "convergence_x": float,
        "convergence_y": float,
        "scale": float (mm/px, optional),
        "case_id": "...",
        "image_name": "..."
      }
    """
    data = request.get_json(silent=True) or {}
    droplets     = data.get("droplets", [])
    conv_x       = data.get("convergence_x")
    conv_y       = data.get("convergence_y")
    scale        = float(data.get("scale", 1.0))
    case_id      = data.get("case_id", "")
    img_name     = data.get("image_name", "")

    if not droplets:
        return jsonify({"error": "No droplet data provided."}), 400
    if conv_x is None or conv_y is None:
        return jsonify({"error": "convergence_x and convergence_y are required."}), 400

    result = compute_3d_origin(droplets, float(conv_x), float(conv_y), scale)

    if case_id:
        _get_analysis_model().save(case_id, img_name, {"reconstruction_3d": result})

    return jsonify(result)
