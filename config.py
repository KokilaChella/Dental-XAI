"""
Central path configuration for the Trust-Aware Dental XAI project.

Every script in this project imports paths from here instead of hardcoding
them. To run locally instead of on Kaggle, change ENVIRONMENT below (or set
the DENTEX_ENV environment variable) and update _DATASET_ROOT for local.

This is the ONLY file that should ever contain a hardcoded absolute path.
"""
import os
from pathlib import Path

# "kaggle" or "local" -- this is the one line you change when switching machines
ENVIRONMENT = os.environ.get("DENTEX_ENV", "kaggle")

if ENVIRONMENT == "kaggle":
    _DATASET_ROOT = Path(
        "/kaggle/input/datasets/truthisneverlinear/dentex-challenge-2023"
    )
    WORKING_DIR = Path("/kaggle/working")

elif ENVIRONMENT == "local":
    # EDIT: point this at wherever you've downloaded the DENTEX dataset
    # (e.g. via `kaggle datasets download truthisneverlinear/dentex-challenge-2023`)
    _DATASET_ROOT = Path("./data/dentex-challenge-2023")
    WORKING_DIR = Path("./outputs")

else:
    raise ValueError(f"Unknown ENVIRONMENT: {ENVIRONMENT!r} (expected 'kaggle' or 'local')")

# ---- dataset inputs (read-only, never written to) ----
JSON_PATH = (
    _DATASET_ROOT
    / "training_data/training_data/quadrant-enumeration-disease"
    / "train_quadrant_enumeration_disease.json"
)
IMAGES_DIR = (
    _DATASET_ROOT
    / "training_data/training_data/quadrant-enumeration-disease/xrays"
)

# ---- project outputs (created and written to by our scripts) ----
SPLITS_DIR = WORKING_DIR / "splits"
LABELS_DIR = WORKING_DIR / "labels"
CROPS_DIR = WORKING_DIR / "crops"

for _d in (SPLITS_DIR, LABELS_DIR, CROPS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
