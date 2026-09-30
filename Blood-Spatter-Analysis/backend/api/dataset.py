"""Dataset API — Module 1 endpoints."""

import os
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from backend.config import Config
from backend.ml.dataset_ingest import ingest_zip, ingest_folder, get_dataset_stats
from backend.database.models import DatasetModel, get_db
from backend.utils.logger import get_logger

log = get_logger(__name__)
dataset_bp = Blueprint("dataset", __name__, url_prefix="/api/dataset")


def _get_model():
    db = get_db(Config.DB_PATH)
    return DatasetModel(db)


@dataset_bp.route("/list", methods=["GET"])
def list_datasets():
    root = Config.DATASET_FOLDER
    datasets = []
    if root.exists():
        for d in sorted(root.iterdir()):
            if d.is_dir() and not d.name.startswith("_"):
                meta_path = d / "metadata.json"
                if meta_path.exists():
                    import json
                    meta = json.loads(meta_path.read_text())
                    datasets.append({"name": d.name, "path": str(d), **meta})
    return jsonify({"datasets": datasets, "count": len(datasets)})


@dataset_bp.route("/statistics/<dataset_name>", methods=["GET"])
def dataset_statistics(dataset_name):
    ds_path = Config.DATASET_FOLDER / secure_filename(dataset_name)
    stats = get_dataset_stats(ds_path)
    return jsonify(stats)


@dataset_bp.route("/upload", methods=["POST"])
def upload_dataset():
    """Upload a ZIP file containing dataset images."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400
    f = request.files["file"]
    if f.filename == "" or not f.filename.endswith(".zip"):
        return jsonify({"error": "Please upload a .zip file."}), 400

    dataset_name = request.form.get("dataset_name", "").strip()
    if not dataset_name:
        dataset_name = Path(secure_filename(f.filename)).stem
    dataset_name = secure_filename(dataset_name)

    seed = int(request.form.get("seed", 42))

    # Save ZIP
    tmp_path = Config.UPLOAD_FOLDER / f"_dataset_upload_{dataset_name}.zip"
    f.save(str(tmp_path))

    try:
        result = ingest_zip(tmp_path, Config.DATASET_FOLDER, dataset_name, seed=seed)
    except Exception as e:
        log.exception("Dataset ingest failed")
        return jsonify({"error": f"Ingestion failed: {str(e)}"}), 500
    finally:
        tmp_path.unlink(missing_ok=True)

    if result.get("status") == "error":
        return jsonify(result), 400

    return jsonify(result), 201


@dataset_bp.route("/ingest", methods=["POST"])
def ingest_local():
    """Ingest from a local folder path (server-side path)."""
    data = request.get_json(silent=True) or {}
    source_path = data.get("source_path", "").strip()
    dataset_name = secure_filename(data.get("dataset_name", "").strip())
    seed = int(data.get("seed", 42))

    if not source_path or not dataset_name:
        return jsonify({"error": "source_path and dataset_name are required."}), 400

    src = Path(source_path)
    if not src.exists() or not src.is_dir():
        return jsonify({"error": f"Source directory not found: {source_path}"}), 400

    try:
        result = ingest_folder(src, Config.DATASET_FOLDER, dataset_name, seed=seed)
    except Exception as e:
        log.exception("Ingest failed")
        return jsonify({"error": str(e)}), 500

    if result.get("status") == "error":
        return jsonify(result), 400
    return jsonify(result), 201


@dataset_bp.route("/delete/<dataset_name>", methods=["DELETE"])
def delete_dataset(dataset_name):
    import shutil
    ds_path = Config.DATASET_FOLDER / secure_filename(dataset_name)
    if not ds_path.exists():
        return jsonify({"error": "Dataset not found."}), 404
    shutil.rmtree(ds_path)
    log.info(f"Dataset deleted: {dataset_name}")
    return jsonify({"message": f"Dataset '{dataset_name}' deleted."})
