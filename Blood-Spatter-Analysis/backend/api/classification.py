"""Classification API — Module 2 endpoints."""

import os
import threading
from pathlib import Path
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from backend.config import Config
from backend.ml.predict import predict
from backend.ml.train import train_model
from backend.ml.model import clear_model_cache
from backend.database.models import AnalysisModel, CaseModel, get_db
from backend.utils.file_utils import allowed_file
from backend.utils.logger import get_logger

log = get_logger(__name__)
classification_bp = Blueprint("classification", __name__, url_prefix="/api")

# Training state (in-memory for simplicity)
_training_state: dict = {
    "running": False,
    "progress": [],
    "result": None,
    "error": None,
}


def _get_analysis_model():
    db = get_db(Config.DB_PATH)
    return AnalysisModel(db)


# ---- Upload & Classify ----

@classification_bp.route("/classify", methods=["POST"])
def classify_image():
    case_id    = request.form.get("case_id", "")
    image_name = request.form.get("image_name", "")

    # Accept either uploaded file or existing path
    if "file" in request.files:
        f = request.files["file"]
        if not f or not allowed_file(f.filename):
            return jsonify({"error": "Invalid or unsupported file format."}), 400
        fname = secure_filename(f.filename)
        save_path = Config.UPLOAD_FOLDER / (case_id or "general") / fname
        save_path.parent.mkdir(parents=True, exist_ok=True)
        f.save(str(save_path))
        image_name = fname
    elif image_name:
        save_path = Config.UPLOAD_FOLDER / (case_id or "general") / secure_filename(image_name)
        if not save_path.exists():
            return jsonify({"error": f"Image not found: {image_name}"}), 404
    else:
        return jsonify({"error": "Provide a file upload or image_name."}), 400

    model_path = Config.MODEL_FOLDER / "bpa_classifier.pth"

    try:
        result = predict(save_path, model_path)
    except FileNotFoundError as e:
        return jsonify({
            "error": "model_unavailable",
            "message": str(e),
            "advice": "Train the model first via the Training page (/training.html).",
        }), 503
    except ValueError as e:
        return jsonify({"error": "invalid_image", "message": str(e)}), 400
    except Exception as e:
        log.exception("Classification failed")
        return jsonify({"error": "classification_failed", "message": str(e)}), 500

    result["image_name"] = image_name
    result["case_id"] = case_id

    # Persist
    if case_id:
        _get_analysis_model().save(case_id, image_name, {"classification": result})

    return jsonify(result)


# ---- Image Upload (just upload, no classify) ----

@classification_bp.route("/upload", methods=["POST"])
def upload_image():
    case_id = request.form.get("case_id", "general")
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400
    f = request.files["file"]
    if not f or not allowed_file(f.filename):
        return jsonify({"error": "Invalid file format."}), 400

    fname = secure_filename(f.filename)
    save_dir = Config.UPLOAD_FOLDER / case_id
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / fname
    f.save(str(save_path))

    # Update case
    if case_id != "general":
        db = get_db(Config.DB_PATH)
        cm = CaseModel(db)
        case = cm.get_by_id(case_id)
        if case:
            images = case.get("images", [])
            if fname not in images:
                images.append(fname)
            cm.update(case_id, {"images": images})

    return jsonify({"filename": fname, "path": str(save_path), "case_id": case_id}), 201


# ---- Model Status ----

@classification_bp.route("/model/status", methods=["GET"])
def model_status():
    model_path = Config.MODEL_FOLDER / "bpa_classifier.pth"
    return jsonify({
        "trained":     model_path.exists(),
        "path":        str(model_path) if model_path.exists() else None,
        "training_running": _training_state["running"],
        "device":      "cuda" if __import__("torch").cuda.is_available() else "cpu",
    })


# ---- Training ----

@classification_bp.route("/model/train", methods=["POST"])
def start_training():
    global _training_state
    if _training_state["running"]:
        return jsonify({"error": "Training already in progress."}), 409

    data = request.get_json(silent=True) or {}
    dataset_name = data.get("dataset_name", "").strip()
    if not dataset_name:
        return jsonify({"error": "dataset_name is required."}), 400

    ds_path = Config.DATASET_FOLDER / secure_filename(dataset_name) / "train"
    if not ds_path.exists():
        return jsonify({"error": f"Training data not found at: {ds_path}"}), 400

    epochs     = int(data.get("epochs", Config.DEFAULT_EPOCHS))
    batch_size = int(data.get("batch_size", Config.DEFAULT_BATCH_SIZE))
    lr         = float(data.get("lr", Config.DEFAULT_LR))
    image_size = int(data.get("image_size", Config.DEFAULT_IMAGE_SIZE))
    seed       = int(data.get("seed", Config.DEFAULT_SEED))

    _training_state = {"running": True, "progress": [], "result": None, "error": None}

    def run_training():
        global _training_state
        def cb(prog):
            _training_state["progress"].append(prog)
        try:
            result = train_model(
                dataset_path=ds_path,
                model_save_path=Config.MODEL_FOLDER / "bpa_classifier.pth",
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                image_size=image_size,
                seed=seed,
                progress_callback=cb,
            )
            _training_state["result"] = result
        except Exception as e:
            log.exception("Training thread error")
            _training_state["error"] = str(e)
        finally:
            _training_state["running"] = False

    t = threading.Thread(target=run_training, daemon=True)
    t.start()
    return jsonify({"message": "Training started.", "status": "running"}), 202


@classification_bp.route("/model/train/progress", methods=["GET"])
def training_progress():
    return jsonify({
        "running":  _training_state["running"],
        "progress": _training_state["progress"],
        "result":   _training_state["result"],
        "error":    _training_state["error"],
    })
