import os
import re
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np
from PIL import Image

from backend.database import Session, Patient, Screening, Clinician
from backend import models as db
import tensorflow as tf
from tensorflow import keras

from backend.config import MODEL_PATH, THRESHOLD, UPLOAD_DIR, GAPCAM_DIR, REPORT_DIR, _BASE

# Application display timezone: Asia/Kolkata (IST, UTC+05:30)
APP_TZ = ZoneInfo("Asia/Kolkata")


def _fmt_ist(dt) -> str:
    """
    Format a database timestamp for the API response.
    - Normalizes naive timestamps (legacy data, stored as UTC) to UTC-aware
    - Converts to Asia/Kolkata and returns ISO 8601 with offset, e.g. '2026-08-31T19:30:00+05:30'
    """
    if dt is None:
        return None
    # SQLite DATETIME strips timezone info; assume all stored values are UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ist_dt = dt.astimezone(APP_TZ)
    return ist_dt.isoformat()

app = FastAPI(title="SIH-2026 DR Screening", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GAPCAM_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(os.path.join(_BASE, "templates"), exist_ok=True)

app.mount("/static", StaticFiles(directory=os.path.join(_BASE, "static")), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# ── Model loading ─────────────────────────────────────────────────────────────

_model = None
_model_dual = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        _model = keras.models.load_model(MODEL_PATH, safe_mode=False)
    return _model


def get_dual_model():
    global _model_dual
    if _model_dual is None:
        # Resolve relative to the project root (parent of app/)
        project_root = os.path.dirname(_BASE)
        dual_path = os.path.join(project_root, "ml_training", "models", "full_training", "dual_output_model.keras")
        if os.path.exists(dual_path):
            _model_dual = keras.models.load_model(dual_path, safe_mode=False)
        else:
            _model_dual = None
    return _model_dual


# ── Inference ─────────────────────────────────────────────────────────────────

def preprocess_image(file_path: str):
    img = Image.open(file_path).convert("RGB")
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def infer_image(file_path: str):
    img = preprocess_image(file_path)
    model = get_model()
    pred = float(model.predict(img, verbose=0)[0][0])
    confidence = pred if pred >= 0.5 else 1.0 - pred
    prediction = int(pred >= THRESHOLD)
    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probability": round(pred, 4),
        "threshold": THRESHOLD,
        "interpretation": "DR Present — Refer to ophthalmologist" if prediction == 1 else "No DR Detected"
    }


# ── Safe path helpers ─────────────────────────────────────────────────────────

def resolve_upload_path(stored_path: str) -> str | None:
    """
    Resolve a stored image path to an absolute path that exists on disk.
    Handles Windows and Unix paths, relative and absolute.
    """
    if not stored_path:
        return None

    candidates = []
    # 1. Use as-is if absolute
    if os.path.isabs(stored_path):
        candidates.append(stored_path)
    else:
        # 2. Relative to _BASE (app/)
        candidates.append(os.path.join(_BASE, stored_path.replace("/", os.sep)))
        # 3. Relative to project root
        candidates.append(os.path.join(os.path.dirname(_BASE), stored_path.replace("/", os.sep)))

    for path in candidates:
        if os.path.exists(path) and os.path.isfile(path):
            return path
    return None


def serve_image_if_exists(path: str | None):
    """Return a FileResponse if the path resolves to an existing file."""
    if not path:
        raise HTTPException(status_code=404, detail="Image path not set")
    resolved = resolve_upload_path(path)
    if not resolved or not os.path.exists(resolved):
        raise HTTPException(status_code=404, detail="Image file not found on disk")
    return FileResponse(resolved)


# ── Clinician auth / profile endpoints ───────────────────────────────────────

# Default clinician ID — single-user app
DEFAULT_CLINICIAN_ID = "CL-000001"


@app.get("/api/clinician")
def api_clinician_profile():
    """Get profile of the currently logged-in clinician."""
    clinician = db.get_clinician(DEFAULT_CLINICIAN_ID)
    if not clinician:
        raise HTTPException(status_code=404, detail="Clinician not found")
    return db.clinician_profile_response(clinician)


@app.put("/api/clinician")
def api_update_clinician(
        name: str = Form(None),
        email: str = Form(None),
        role: str = Form(None)):
    """Update clinician profile fields."""
    clinician = db.get_clinician(DEFAULT_CLINICIAN_ID)
    if not clinician:
        raise HTTPException(status_code=404, detail="Clinician not found")

    # Validate email format if provided
    if email is not None:
        email = email.strip().lower()
        if not re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", email):
            raise HTTPException(status_code=400, detail="Invalid email address")
        # Check for duplicate (excluding current clinician)
        existing = db.get_clinician_by_email(email)
        if existing and existing.clinician_id != DEFAULT_CLINICIAN_ID:
            raise HTTPException(status_code=409, detail="Email already in use by another account")

    if name is not None:
        name = name.strip()
        if len(name) < 2:
            raise HTTPException(status_code=400, detail="Name must be at least 2 characters")

    # Build update dict (only non-None fields)
    fields = {}
    if name is not None:
        fields["name"] = name
    if email is not None:
        fields["email"] = email
    # role is intentionally read-only for safety (only admin can change roles)

    if not fields:
        raise HTTPException(status_code=400, detail="No fields provided to update")

    updated = db.update_clinician(DEFAULT_CLINICIAN_ID, **fields)
    return {"ok": True, "clinician": db.clinician_profile_response(updated)}


@app.post("/api/auth/change-password")
def api_change_password(
        current_password: str = Form(...),
        new_password: str = Form(...),
        confirm_password: str = Form(...)):
    """Change the clinician's password. Passwords are hashed with bcrypt."""
    import bcrypt

    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")
    if new_password != confirm_password:
        raise HTTPException(status_code=400, detail="New password and confirmation do not match")
    if new_password == current_password:
        raise HTTPException(status_code=400, detail="New password must be different from current password")

    clinician = db.get_clinician(DEFAULT_CLINICIAN_ID)
    if not clinician:
        raise HTTPException(status_code=404, detail="Clinician not found")

    # Verify current password
    stored_hash = clinician.password_hash
    current_hash = bcrypt.hashpw(current_password.encode(), bcrypt.gensalt())
    if stored_hash:
        if not bcrypt.checkpw(current_password.encode(), stored_hash.encode()):
            raise HTTPException(status_code=401, detail="Current password is incorrect")
    else:
        # No hash stored — demo account, still verify against default
        if current_password != "password":
            raise HTTPException(status_code=401, detail="Current password is incorrect")

    # Hash and store new password
    new_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode("utf-8")
    db.update_clinician(DEFAULT_CLINICIAN_ID, password_hash=new_hash)
    return {"ok": True, "message": "Password changed successfully"}


# ── Patient endpoints ──────────────────────────────────────────────────────────

@app.get("/api/patients")
def api_patients(q: str = ""):
    if q:
        results = db.search_patients(q)
    else:
        results = db.list_patients()
    return [{"patient_id": p.patient_id, "name": p.name, "age": p.age,
             "gender": p.gender, "phone": p.phone, "created_at": _fmt_ist(p.created_at)} for p in results]


@app.post("/api/patients")
def api_create_patient(name: str = Form(...), age: int = Form(...), gender: str = Form(None),
                       phone: str = Form(None), email: str = Form(None), address: str = Form(None)):
    p = db.create_patient(name, age, gender, phone, email, address)
    return {"ok": True, "patient": {"patient_id": p.patient_id, "name": p.name}}


@app.get("/api/patients/{patient_id}")
def api_get_patient(patient_id: str):
    p = db.get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    screenings = db.get_screenings_for_patient(patient_id)
    return {"patient": {"patient_id": p.patient_id, "name": p.name, "age": p.age,
                        "gender": p.gender, "phone": p.phone, "email": p.email,
                        "address": p.address, "created_at": _fmt_ist(p.created_at)},
             "screenings": [{"screening_id": s.screening_id, "date": _fmt_ist(s.screening_date),
                             "prediction": s.prediction, "confidence": s.confidence,
                             "image_path": s.image_path, "gapcam_path": s.gapcam_path} for s in screenings]}


@app.put("/api/patients/{patient_id}")
def api_update_patient(patient_id: str, name: str = Form(None), age: int = Form(None), gender: str = Form(None),
                       phone: str = Form(None), email: str = Form(None), address: str = Form(None)):
    fields = {k: v for k, v in {"name": name, "age": age, "gender": gender, "phone": phone, "email": email, "address": address}.items() if v is not None}
    p = db.update_patient(patient_id, **fields)
    if not p:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True, "patient_id": p.patient_id}


@app.delete("/api/patients/{patient_id}")
def api_delete_patient(patient_id: str):
    ok = db.delete_patient(patient_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True}


# ── Screening / Inference endpoint ────────────────────────────────────────────

@app.post("/api/screenings/predict")
def api_predict(patient_id: str = Form(...), file: UploadFile = File(...)):
    # Validate patient exists
    p = db.get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=400, detail="Patient not found")

    # Validate file
    allowed = {"image/png", "image/jpeg", "image/jpg", "image/tiff"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Invalid image type")

    # Save upload — always store relative to UPLOAD_DIR
    ext = Path(file.filename).suffix or ".jpg"
    filename = f"{patient_id}_{db.screening_count()}_{int(__import__('time').time()*1000)}{ext}"
    # Store as relative path with forward slashes (cross-platform)
    relative_filename = "uploads/" + filename
    image_path_abs = os.path.join(UPLOAD_DIR, filename)

    with open(image_path_abs, "wb") as f:
        f.write(file.file.read())

    # Run inference
    result = infer_image(image_path_abs)

    # Generate GAP-CAM using existing dual_output_model + dense_weights.json
    gapcam_path = None
    try:
        from backend.gapcam import generate_gapcam
        gapcam_filename = filename.replace(ext, "_gapcam.jpg")
        gapcam_path_abs = os.path.join(GAPCAM_DIR, gapcam_filename)
        result_gap = generate_gapcam(image_path_abs, gapcam_path_abs)
        if result_gap:
            # Store relative path with forward slashes (cross-platform)
            gapcam_path = "uploads/gapcam/" + gapcam_filename
        else:
            gapcam_path = None
    except Exception as e:
        print(f"[GAP-CAM] Generation failed: {e}")
        import traceback; traceback.print_exc()
        gapcam_path = None

    # Create screening record — always store relative paths
    screening = db.create_screening(
        patient_id=patient_id,
        prediction=result["probability"],
        confidence=result["confidence"],
        image_path=relative_filename,
        gapcam_path=gapcam_path,
        clinician_notes=None
    )

    return {
        "ok": True,
        "screening_id": screening.screening_id,
        "patient_id": patient_id,
        "prediction": result["prediction"],
        "probability": result["probability"],
        "confidence": result["confidence"],
        "threshold": THRESHOLD,
        "interpretation": result["interpretation"],
        "heatmap_path": gapcam_path,
        "image_path": relative_filename,
        "screening_date": _fmt_ist(screening.screening_date)
    }


# ── Screening list ────────────────────────────────────────────────────────────

@app.get("/api/screenings")
def api_screenings(limit: int = 20):
    screenings = db.list_screenings(limit)
    return [{"screening_id": s.screening_id, "patient_id": s.patient_id,
             "date": _fmt_ist(s.screening_date), "prediction": s.prediction,
             "confidence": s.confidence, "image_path": s.image_path,
             "gapcam_path": s.gapcam_path} for s in screenings]


@app.get("/api/screenings/{screening_id}")
def api_screening_by_id(screening_id: str):
    s = db.get_screening(screening_id)
    if not s:
        raise HTTPException(status_code=404, detail="Screening not found")
    p = db.get_patient(s.patient_id)
    return {
        "screening_id": s.screening_id,
        "patient_id": s.patient_id,
        "patient_name": p.name if p else "",
        "date": _fmt_ist(s.screening_date),
        "prediction": float(s.prediction) if s.prediction is not None else 0,
        "confidence": float(s.confidence) if s.confidence is not None else 0,
        "image_path": s.image_path,
        "gapcam_path": s.gapcam_path,
        "clinician_notes": s.clinician_notes,
        "threshold": THRESHOLD
    }


@app.put("/api/screenings/{screening_id}/notes")
def api_screening_notes(screening_id: str, notes: str = Form("")):
    s = db.update_screening_notes(screening_id, notes)
    if not s:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True, "screening_id": s.screening_id, "notes": s.clinician_notes}


# ── Image endpoints ───────────────────────────────────────────────────────────

@app.get("/api/screenings/{screening_id}/image")
def api_screening_image(screening_id: str):
    """Serve the original retinal image for a screening."""
    s = db.get_screening(screening_id)
    if not s:
        raise HTTPException(status_code=404, detail="Screening not found")
    return serve_image_if_exists(s.image_path)


@app.get("/api/screenings/{screening_id}/heatmap")
def api_screening_heatmap(screening_id: str):
    """
    Serve the GAP-CAM heatmap for a screening.
    If no heatmap exists on disk, regenerate it on-demand.
    """
    s = db.get_screening(screening_id)
    if not s:
        raise HTTPException(status_code=404, detail="Screening not found")

    # Try existing path first
    if s.gapcam_path:
        resolved = resolve_upload_path(s.gapcam_path)
        if resolved and os.path.exists(resolved):
            return FileResponse(resolved)

    # Regenerate GAP-CAM if missing or file not found
    if not s.image_path:
        raise HTTPException(status_code=404, detail="Original image not available for heatmap generation")

    resolved_img = resolve_upload_path(s.image_path)
    if not resolved_img or not os.path.exists(resolved_img):
        raise HTTPException(status_code=404, detail="Original image file not found on disk")

    try:
        from backend.gapcam import generate_gapcam
        import hashlib
        # Generate a unique heatmap filename from screening ID
        heatmap_basename = f"heatmap_{screening_id}.jpg"
        heatmap_path_abs = os.path.join(GAPCAM_DIR, heatmap_basename)
        result = generate_gapcam(resolved_img, heatmap_path_abs)
        if result and os.path.exists(result):
            # Update DB with new relative path
            relative_gapcam = os.path.join("uploads", "gapcam", heatmap_basename).replace("\\", "/")
            _update_screening_gapcam_path(screening_id, relative_gapcam)
            return FileResponse(result)
        else:
            raise HTTPException(status_code=500, detail="GAP-CAM generation failed")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[GAP-CAM] Error: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail="GAP-CAM generation error")


def _update_screening_gapcam_path(screening_id: str, gapcam_path: str):
    """Helper to update screening's gapcam_path in DB."""
    session = db.Session()
    s = session.query(db.Screening).filter_by(screening_id=screening_id).first()
    if s:
        s.gapcam_path = gapcam_path
        session.commit()
    session.close()


# ── Report endpoint ───────────────────────────────────────────────────────────

@app.get("/api/reports/{screening_id}")
def api_report(screening_id: str):
    from backend.reports import generate_report
    screening = db.get_screening(screening_id)
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")
    patient = db.get_patient(screening.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    pdata = {
        "patient_id": patient.patient_id, "name": patient.name,
        "age": patient.age, "gender": patient.gender,
        "phone": patient.phone, "email": patient.email,
        "address": patient.address, "created_at": _fmt_ist(patient.created_at)
    }
    sdata = {
        "screening_id": screening.screening_id,
        "screening_date": _fmt_ist(screening.screening_date),
        "prediction": float(screening.prediction),
        "confidence": float(screening.confidence),
        "image_path": screening.image_path,
        "gapcam_path": screening.gapcam_path,
        "clinician_notes": screening.clinician_notes
    }
    path = generate_report(screening_id, pdata, sdata)
    return FileResponse(path, media_type="text/html", filename=f"report_{screening_id}.html")


# ── Summary ──────────────────────────────────────────────────────────────────

@app.get("/api/summary")
def summary():
    return {
        "patients": db.patient_count(),
        "screenings": db.screening_count(),
        "recent": len(db.list_screenings(5)),
        "model": "EfficientNetV2B0 (INT8)",
        "version": "v1.0.0"
    }


# ── Root / UI ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    index_path = os.path.join(_BASE, "templates", "index.html")
    return HTMLResponse(open(index_path).read())
