import os
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_URL = f"sqlite:///{_BASE}/database/dr_screening.db"
MODEL_PATH = f"{os.path.dirname(_BASE)}/ml_training/models/full_training/best_model.keras"
DUAL_MODEL_PATH = f"{os.path.dirname(_BASE)}/ml_training/models/full_training/dual_output_model.keras"
DENSE_WEIGHTS = f"{os.path.dirname(_BASE)}/ml_training/android_model/dense_weights.json"
UPLOAD_DIR = f"{_BASE}/uploads"
GAPCAM_DIR = f"{_BASE}/uploads/gapcam"
REPORT_DIR = f"{_BASE}/reports"
THRESHOLD = 0.5
