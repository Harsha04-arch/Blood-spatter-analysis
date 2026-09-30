"""
Database models using TinyDB (lightweight JSON database).
No external database server required.
"""

import uuid
from datetime import datetime
from pathlib import Path
from tinydb import TinyDB, Query
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware

_db_instance = None


def get_db(db_path: Path) -> TinyDB:
    """Return the singleton TinyDB instance."""
    global _db_instance
    if _db_instance is None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _db_instance = TinyDB(str(db_path))
    return _db_instance


# --------------------------------------------------------------------------- #
#  Helper                                                                      #
# --------------------------------------------------------------------------- #

def new_id() -> str:
    return str(uuid.uuid4())


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


# --------------------------------------------------------------------------- #
#  Cases                                                                       #
# --------------------------------------------------------------------------- #

class CaseModel:
    TABLE = "cases"

    def __init__(self, db: TinyDB):
        self.table = db.table(self.TABLE)

    def create(self, name: str, description: str = "") -> dict:
        doc = {
            "id": new_id(),
            "name": name,
            "description": description,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "images": [],
            "classification_results": [],
            "droplet_analysis": [],
            "convergence_2d": None,
            "reconstruction_3d": None,
            "report_path": None,
        }
        self.table.insert(doc)
        return doc

    def get_all(self) -> list:
        return self.table.all()

    def get_by_id(self, case_id: str) -> dict | None:
        Q = Query()
        results = self.table.search(Q.id == case_id)
        return results[0] if results else None

    def update(self, case_id: str, data: dict) -> bool:
        Q = Query()
        data["updated_at"] = now_iso()
        return bool(self.table.update(data, Q.id == case_id))

    def delete(self, case_id: str) -> bool:
        Q = Query()
        return bool(self.table.remove(Q.id == case_id))


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #

class DatasetModel:
    TABLE = "datasets"

    def __init__(self, db: TinyDB):
        self.table = db.table(self.TABLE)

    def create(self, name: str, path: str, metadata: dict) -> dict:
        doc = {
            "id": new_id(),
            "name": name,
            "path": path,
            "created_at": now_iso(),
            **metadata,
        }
        self.table.insert(doc)
        return doc

    def get_all(self) -> list:
        return self.table.all()

    def get_by_name(self, name: str) -> dict | None:
        Q = Query()
        results = self.table.search(Q.name == name)
        return results[0] if results else None

    def delete(self, dataset_id: str) -> bool:
        Q = Query()
        return bool(self.table.remove(Q.id == dataset_id))


# --------------------------------------------------------------------------- #
#  Analysis Results                                                            #
# --------------------------------------------------------------------------- #

class AnalysisModel:
    TABLE = "analyses"

    def __init__(self, db: TinyDB):
        self.table = db.table(self.TABLE)

    def save(self, case_id: str, image_name: str, result: dict) -> dict:
        Q = Query()
        existing = self.table.search((Q.case_id == case_id) & (Q.image_name == image_name))
        doc = {
            "id": new_id(),
            "case_id": case_id,
            "image_name": image_name,
            "created_at": now_iso(),
            **result,
        }
        if existing:
            doc["id"] = existing[0]["id"]
            self.table.update(doc, (Q.case_id == case_id) & (Q.image_name == image_name))
        else:
            self.table.insert(doc)
        return doc

    def get_by_case(self, case_id: str) -> list:
        Q = Query()
        return self.table.search(Q.case_id == case_id)

    def get_latest(self, case_id: str, image_name: str) -> dict | None:
        Q = Query()
        results = self.table.search((Q.case_id == case_id) & (Q.image_name == image_name))
        return results[-1] if results else None
