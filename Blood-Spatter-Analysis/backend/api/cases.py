"""Cases API — CRUD for forensic cases."""

from flask import Blueprint, request, jsonify
from backend.database.models import CaseModel, get_db
from backend.config import Config
from backend.utils.logger import get_logger

log = get_logger(__name__)
cases_bp = Blueprint("cases", __name__, url_prefix="/api/cases")


def _get_model():
    db = get_db(Config.DB_PATH)
    return CaseModel(db)


@cases_bp.route("", methods=["GET"])
def list_cases():
    model = _get_model()
    cases = model.get_all()
    return jsonify({"cases": cases, "count": len(cases)})


@cases_bp.route("", methods=["POST"])
def create_case():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Case name is required."}), 400
    desc = data.get("description", "")
    model = _get_model()
    case = model.create(name, desc)
    log.info(f"Case created: {case['id']} — {name}")
    return jsonify(case), 201


@cases_bp.route("/<case_id>", methods=["GET"])
def get_case(case_id):
    model = _get_model()
    case = model.get_by_id(case_id)
    if not case:
        return jsonify({"error": "Case not found."}), 404
    return jsonify(case)


@cases_bp.route("/<case_id>", methods=["PUT"])
def update_case(case_id):
    data = request.get_json(silent=True) or {}
    allowed = {"name", "description"}
    update_data = {k: v for k, v in data.items() if k in allowed}
    if not update_data:
        return jsonify({"error": "No valid fields to update."}), 400
    model = _get_model()
    if not model.get_by_id(case_id):
        return jsonify({"error": "Case not found."}), 404
    model.update(case_id, update_data)
    return jsonify(model.get_by_id(case_id))


@cases_bp.route("/<case_id>", methods=["DELETE"])
def delete_case(case_id):
    model = _get_model()
    if not model.get_by_id(case_id):
        return jsonify({"error": "Case not found."}), 404
    model.delete(case_id)
    log.info(f"Case deleted: {case_id}")
    return jsonify({"message": "Case deleted."}), 200
