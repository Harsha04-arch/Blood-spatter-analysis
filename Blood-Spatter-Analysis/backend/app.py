"""
BPA Application — Flask Entry Point

Deep Learning-Based Bloodstain Pattern Analysis
for Crime Scene Analysis and Origin Estimation
and Weapon Category Estimation.
"""

import sys
from pathlib import Path

# ── Ensure backend package is importable ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ─────────────────────────────────────────────────────────────────────────────

from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS

from backend.config import Config
from backend.api.cases import cases_bp
from backend.api.dataset import dataset_bp
from backend.api.classification import classification_bp
from backend.api.droplets import droplets_bp
from backend.api.convergence import convergence_bp
from backend.api.reports import reports_bp
from backend.utils.logger import get_logger

log = get_logger("bpa.app")


def create_app() -> Flask:
    Config.ensure_dirs()

    app = Flask(
        __name__,
        static_folder=str(ROOT / "frontend"),
        static_url_path="/frontend",
    )
    app.config["SECRET_KEY"] = Config.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register blueprints
    for bp in [cases_bp, dataset_bp, classification_bp, droplets_bp, convergence_bp, reports_bp]:
        app.register_blueprint(bp)

    # ── Serve frontend ─────────────────────────────────────────────────────────
    @app.route("/")
    def index():
        return send_from_directory(str(ROOT / "frontend"), "index.html")

    @app.route("/<path:filename>")
    def serve_frontend(filename):
        frontend_dir = ROOT / "frontend"
        target = frontend_dir / filename
        if target.exists() and target.is_file():
            return send_from_directory(str(frontend_dir), filename)
        # SPA fallback
        return send_from_directory(str(frontend_dir), "index.html")

    # ── Health check ───────────────────────────────────────────────────────────
    @app.route("/api/health")
    def health():
        import torch
        model_path = Config.MODEL_FOLDER / "bpa_classifier.pth"
        return jsonify({
            "status": "ok",
            "model_trained": model_path.exists(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "pytorch_version": torch.__version__,
        })

    # ── Global error handlers ──────────────────────────────────────────────────
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": "Bad request", "message": str(e)}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"error": "File too large. Maximum upload size is 50 MB."}), 413

    @app.errorhandler(500)
    def server_error(e):
        log.exception("Unhandled server error")
        return jsonify({"error": "Internal server error", "message": str(e)}), 500

    log.info("BPA application created successfully.")
    return app


if __name__ == "__main__":
    app = create_app()
    log.info("Starting BPA server at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=Config.DEBUG, use_reloader=False)
