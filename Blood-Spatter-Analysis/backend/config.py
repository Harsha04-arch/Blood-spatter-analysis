"""
BPA Application Configuration
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "bpa-dev-secret-key-change-in-production")
    FLASK_ENV = os.environ.get("FLASK_ENV", "development")
    DEBUG = os.environ.get("FLASK_DEBUG", "True").lower() == "true"

    # Paths
    UPLOAD_FOLDER = BASE_DIR / os.environ.get("UPLOAD_FOLDER", "uploads")
    DATASET_FOLDER = BASE_DIR / os.environ.get("DATASET_FOLDER", "datasets")
    MODEL_FOLDER = BASE_DIR / os.environ.get("MODEL_FOLDER", "models")
    REPORTS_FOLDER = BASE_DIR / os.environ.get("REPORTS_FOLDER", "reports")
    DB_PATH = BASE_DIR / os.environ.get("DB_PATH", "database/bpa.json")

    # Upload limits
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_UPLOAD_MB", 50)) * 1024 * 1024

    # Allowed image extensions
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp"}

    # Dataset classes
    SPATTER_CLASSES = [
        "Passive Drip",
        "Transfer Smear",
        "Medium Velocity Impact Spatter",
        "High Velocity Impact Spatter",
    ]

    # Velocity classes
    VELOCITY_CLASSES = [
        "Passive / Low Velocity",
        "Medium Velocity",
        "High Velocity",
    ]

    # Default training hyperparameters
    DEFAULT_EPOCHS = 20
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_LR = 1e-4
    DEFAULT_IMAGE_SIZE = 224
    DEFAULT_SEED = 42

    # CV defaults
    DEFAULT_MIN_DROPLET_AREA = 50
    DEFAULT_MAX_DROPLET_AREA = 50000
    DEFAULT_THRESHOLD = 127
    DEFAULT_KERNEL_SIZE = 3

    @staticmethod
    def ensure_dirs():
        for attr in ["UPLOAD_FOLDER", "DATASET_FOLDER", "MODEL_FOLDER", "REPORTS_FOLDER"]:
            path = getattr(Config, attr)
            path.mkdir(parents=True, exist_ok=True)
        Config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
