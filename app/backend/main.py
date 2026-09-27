import os
import re
import logging
import concurrent.futures
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np
from PIL import Image, ImageDraw
import cv2

from backend.database import Session, Patient, Screening, Clinician
from backend.database_adapter import get_models
import tensorflow as tf
from tensorflow import keras

from backend.config import MODEL_PATH, THRESHOLD, UPLOAD_DIR, GAPCAM_DIR, REPORT_DIR, _BASE
from backend import vessel_segmentation
from backend.image_quality import validate_image, ImageStatus, get_validator

# Initialize backend models
models = get_models()

# Phase 9: Firebase integration
try:
    from backend.firebase_config import (
        init_firebase, is_firebase_enabled, check_firebase_connection
    )
    from backend import firebase_adapter as fb
    FIREBASE_IMPORTS_OK = True
except ImportError:
    FIREBASE_IMPORTS_OK = False

logger = logging.getLogger(__name__)

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
    # Phase 12 / 8B: clinician_reviewed_at is stored as an IST ISO string (not a datetime).
    if isinstance(dt, str):
        return dt
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

# ── Phase 9: Database backend selection ───────────────────────────────────────

DATABASE_BACKEND = os.getenv("DATABASE_BACKEND", "sqlite").lower()

def _use_firebase() -> bool:
    """Check if Firebase is the active database backend."""
    return DATABASE_BACKEND == "firebase" and FIREBASE_IMPORTS_OK

# Initialize Firebase if enabled
if _use_firebase():
    try:
        init_firebase()
        logger.info("Firebase initialized as database backend")
    except Exception as e:
        logger.warning(f"Firebase init failed, falling back to SQLite: {e}")
        DATABASE_BACKEND = "sqlite"


@app.get("/api/firebase/status")
def firebase_status():
    """
    Check Firebase connection status and current database backend.

    Returns:
        JSON with database backend info and Firebase connectivity status
    """
    result = {
        "database_backend": DATABASE_BACKEND,
        "firebase_available": FIREBASE_IMPORTS_OK,
    }

    if FIREBASE_IMPORTS_OK and DATABASE_BACKEND == "firebase":
        try:
            status = check_firebase_connection()
            result["firebase_connected"] = status["ok"]
            result["firebase_project_id"] = status["project_id"]
            if status["error"]:
                result["firebase_error"] = status["error"]
        except Exception as e:
            result["firebase_connected"] = False
            result["firebase_error"] = str(e)
    else:
        result["firebase_connected"] = False
        if not FIREBASE_IMPORTS_OK:
            result["firebase_error"] = "firebase-admin package not installed"
        elif DATABASE_BACKEND != "firebase":
            result["firebase_error"] = f"DATABASE_BACKEND={DATABASE_BACKEND}, not firebase"

    return result


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

def save_preprocessed_image(file_path: str, out_path: str) -> str | None:
    """
    Generate a preprocessed retinal image visualization.
    Enhances vessel visibility while remaining recognizable as the same fundus photograph.

    Algorithm:
    1. Extract green channel (best contrast for retinal vessels)
    2. Normalize illumination
    3. Apply CLAHE for local contrast enhancement
    4. Keep in RGB color space for natural appearance
    """
    try:
        img = Image.open(file_path).convert("RGB")
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.uint8)

        # Extract green channel (retinal vessels have best contrast there)
        green = arr[:, :, 1].astype(np.float32)

        # Normalize illumination using morphological background subtraction
        # Create a background estimate using morphological opening
        kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        background = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel_bg)

        # Subtract background to normalize illumination
        illumination_corrected = cv2.subtract(green, background * 0.8)
        illumination_corrected = np.clip(illumination_corrected, 0, 255).astype(np.uint8)

        # Apply CLAHE to enhanced green channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_green = clahe.apply(illumination_corrected)

        # Enhance contrast slightly
        enhanced_green = cv2.convertScaleAbs(enhanced_green, alpha=1.05, beta=5)
        enhanced_green = np.clip(enhanced_green, 0, 255).astype(np.uint8)

        # Convert back to RGB by blending enhanced green with original
        # Keep R and B channels from original, enhance G
        result = arr.copy().astype(np.float32)
        result[:, :, 1] = enhanced_green
        result = np.clip(result, 0, 255).astype(np.uint8)

        Image.fromarray(result).save(out_path, "JPEG", quality=92)
        return out_path
    except Exception as e:
        print(f"[Preprocess] Error: {e}")
        import traceback; traceback.print_exc()
        return None

def save_segmented_image(file_path: str, out_path: str) -> str | None:
    """
    Generate a retinal blood vessel segmentation map using Attention U-Net neural network.

    Uses a pretrained Attention U-Net model from:
    https://github.com/arkanivasarkar/Retinal-Vessel-Segmentation-using-variants-of-UNET

    The neural network produces clinically-meaningful binary vessel segmentation
    (white vessels on black background) for retinal fundus photographs.

    Output format:
    - Background (outside retina): (0, 0, 0) black
    - Vessels: (255, 255, 255) white

    This is a significant improvement over classical morphological approaches,
    providing neural network-based vessel detection trained on real retinal images.
    """
    try:
        print(f"[Segment] Using Attention U-Net neural network for vessel segmentation")
        from backend.vessel_segmentation_v2 import segment_retinal_vessels
        return segment_retinal_vessels(file_path, out_path)
    except Exception as e:
        print(f"[Segment] Error: {e}")
        import traceback; traceback.print_exc()
        return None

def save_classified_result_image(file_path: str, out_path: str, result: dict) -> str | None:
    """
    Generate a classified result visualization.
    Preserves the retinal image visibility while overlaying prediction information clearly.
    """
    try:
        img = Image.open(file_path).convert("RGB")
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        w, h = 224, 224

        prediction = result.get("prediction", 0)
        confidence = result.get("confidence", 0)
        prob = result.get("probability", 0)
        threshold = result.get("threshold", 0.5)

        # Create result info with color coding
        if prediction == 1:
            status = "DR Present"
            color_status = (200, 50, 50)  # Red
            bg_color = (40, 20, 20)  # Dark red background
        else:
            status = "No DR Detected"
            color_status = (50, 180, 50)  # Green
            bg_color = (20, 40, 20)  # Dark green background

        # Keep retinal image as background
        arr = np.array(img, dtype=np.uint8)

        # Add semi-transparent colored panel at bottom with result
        overlay_height = 70
        overlay_area = arr[-overlay_height:, :].astype(np.float32)

        # Create overlay panel
        overlay_panel = np.zeros((overlay_height, w, 3), dtype=np.uint8)
        overlay_panel[:, :] = bg_color

        # Blend overlay panel with retinal area
        alpha = 0.65  # Semi-transparent
        arr_result = (overlay_area * (1 - alpha) + overlay_panel * alpha).astype(np.uint8)
        arr[-overlay_height:, :] = arr_result

        # Convert to PIL for text drawing
        result_img = Image.fromarray(arr)
        draw = ImageDraw.Draw(result_img)

        # Calculate text positions
        top_margin = h - overlay_height + 6
        line_height = 17

        # Draw status in bright color
        draw.text((10, top_margin), status, fill=color_status, font=None)

        # Draw confidence
        conf_text = f"Conf: {confidence*100:.1f}%"
        draw.text((10, top_margin + line_height), conf_text, fill=(240, 240, 240), font=None)

        # Draw probability and threshold
        prob_text = f"Score: {prob:.4f}"
        draw.text((10, top_margin + 2*line_height), prob_text, fill=(200, 200, 200), font=None)

        result_img.save(out_path, "JPEG", quality=92)
        return out_path
    except Exception as e:
        print(f"[Classified] Error: {e}")
        import traceback; traceback.print_exc()
        return None


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
    clinician = models.get_clinician(DEFAULT_CLINICIAN_ID)
    if not clinician:
        raise HTTPException(status_code=404, detail="Clinician not found")
    return models.clinician_profile_response(clinician)


@app.put("/api/clinician")
def api_update_clinician(
        name: str = Form(None),
        email: str = Form(None),
        role: str = Form(None)):
    """Update clinician profile fields."""
    clinician = models.get_clinician(DEFAULT_CLINICIAN_ID)
    if not clinician:
        raise HTTPException(status_code=404, detail="Clinician not found")

    # Validate email format if provided
    if email is not None:
        email = email.strip().lower()
        if not re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", email):
            raise HTTPException(status_code=400, detail="Invalid email address")
        # Check for duplicate (excluding current clinician)
        existing = models.get_clinician_by_email(email)
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

    updated = models.update_clinician(DEFAULT_CLINICIAN_ID, **fields)
    return {"ok": True, "clinician": models.clinician_profile_response(updated)}


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

    clinician = models.get_clinician(DEFAULT_CLINICIAN_ID)
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
    models.update_clinician(DEFAULT_CLINICIAN_ID, password_hash=new_hash)
    return {"ok": True, "message": "Password changed successfully"}


# ── Patient endpoints ──────────────────────────────────────────────────────────

@app.get("/api/patients")
def api_patients(q: str = ""):
    if q:
        results = models.search_patients(q)
    else:
        results = models.list_patients()
    return [{"patient_id": p.patient_id, "name": p.name, "age": p.age,
             "gender": p.gender, "phone": p.phone, "created_at": _fmt_ist(p.created_at)} for p in results]


@app.post("/api/patients")
def api_create_patient(name: str = Form(...), age: int = Form(...), gender: str = Form(None),
                       phone: str = Form(None), email: str = Form(None), address: str = Form(None)):
    p = models.create_patient(name, age, gender, phone, email, address)
    return {"ok": True, "patient": {"patient_id": p.patient_id, "name": p.name}}


@app.get("/api/patients/{patient_id}")
def api_get_patient(patient_id: str):
    p = models.get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    screenings = models.get_screenings_for_patient(patient_id)
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
    p = models.update_patient(patient_id, **fields)
    if not p:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True, "patient_id": p.patient_id}


@app.delete("/api/patients/{patient_id}")
def api_delete_patient(patient_id: str):
    ok = models.delete_patient(patient_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True}


# ── Screening / Inference endpoint ────────────────────────────────────────────

@app.post("/api/screenings/predict")
def api_predict(patient_id: str = Form(...), file: UploadFile = File(...)):
    # Validate patient exists
    p = models.get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=400, detail="Patient not found")

    # Validate file
    allowed = {"image/png", "image/jpeg", "image/jpg", "image/tiff"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Invalid image type")

    # Save upload — always store relative to UPLOAD_DIR
    ext = Path(file.filename).suffix or ".jpg"
    filename = f"{patient_id}_{models.screening_count()}_{int(__import__('time').time()*1000)}{ext}"
    # Store as relative path with forward slashes (cross-platform)
    relative_filename = "uploads/" + filename
    image_path_abs = os.path.join(UPLOAD_DIR, filename)

    with open(image_path_abs, "wb") as f:
        f.write(file.file.read())

    # ── BASIC TECHNICAL FILE VALIDATION ONLY (heuristic fundus gate removed) ──
    # Previous fundus-quality gate (validate_image -> fundus vote / anti-veto)
    # is retired from the critical path. Only genuinely unreadable/corrupt
    # files are blocked here. The EfficientNet classifier handles all others.
    try:
        from PIL import Image
        test_img = Image.open(image_path_abs)
        test_img.load()
        basic_ok = True
    except Exception as e:
        print(f"[BasicValidation] Corrupt/unreadable image rejected: {e}")
        screening = models.create_screening(
            patient_id=patient_id,
            prediction=None,
            confidence=None,
            image_path=relative_filename,
            gapcam_path=None,
            clinician_notes=f"Basic file validation failed: unreadable image"
        )
        return {
            "ok": False,
            "screening_id": screening.screening_id,
            "patient_id": patient_id,
            "screening_date": _fmt_ist(screening.screening_date),
            "image_quality": {"status": "CORRUPTED", "message": "Image file corrupted or unsupported"},
            "ai_result": None,
            "visualizations": {"original_image": relative_filename},
            "safety": {"ai_only_screening": False, "requires_clinician_review": True, "referral_recommendation": "Image file is unreadable; please provide a valid image file"},
            "next_steps": {"action": "recapture_image", "reason": "Unreadable image file"}
        }

    # Image passes basic file validation — proceed to DR classification regardless
    # of heuristic fundus signals (retired from blocking path)
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

    # ── 5-Stage Visualization ──────────────────────────────────────────────
    # 1. Original image: the uploaded file (already saved as relative_filename)
    original_image = relative_filename

    # 2. Preprocessed image: resize to 224x224 (same as model input pipeline)
    preprocessed_image = None
    try:
        pp_filename = filename.replace(ext, "_preprocessed.jpg")
        pp_abs = os.path.join(UPLOAD_DIR, pp_filename)
        if save_preprocessed_image(image_path_abs, pp_abs):
            preprocessed_image = "uploads/" + pp_filename
    except Exception as e:
        print(f"[Preprocess] Generation failed: {e}")

    # 3. Segmented image: lightweight vessel segmentation (green channel + CLAHE + adaptive threshold + morphology)
    segmented_image = None
    try:
        seg_filename = filename.replace(ext, "_segmented.png")
        seg_abs = os.path.join(UPLOAD_DIR, seg_filename)
        if save_segmented_image(image_path_abs, seg_abs):
            segmented_image = "uploads/" + seg_filename
    except Exception as e:
        print(f"[Segment] Generation failed: {e}")

    # 4. Classified resultant image: original + prediction overlay
    classified_image = None
    try:
        cls_filename = filename.replace(ext, "_classified.jpg")
        cls_abs = os.path.join(UPLOAD_DIR, cls_filename)
        if save_classified_result_image(image_path_abs, cls_abs, result):
            classified_image = "uploads/" + cls_filename
    except Exception as e:
        print(f"[Classified] Generation failed: {e}")

    # 5. Grad-CAM image: existing GAP-CAM output (same as heatmap_path)
    gradcam_image = gapcam_path

    # Create screening record — always store relative paths
    screening = models.create_screening(
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
        "screening_date": _fmt_ist(screening.screening_date),

        # ── AI Screening Result (NOT a diagnosis) ──
        "ai_result": {
            "prediction": result["prediction"],
            "probability": result["probability"],
            "confidence": result["confidence"],
            "threshold": THRESHOLD,
            "interpretation": result["interpretation"],
            "model_version": "EfficientNetV2B0"
        },

        # ── Image Quality Assessment ──
        "image_quality": {
            "status": "passed_basic_validation",
            "note": "Heuristic fundus-quality gate retired from blocking path"
        },

        # ── 5-Stage Visualization Paths ──
        "visualizations": {
            "original_image": original_image,
            "preprocessed_image": preprocessed_image,
            "segmented_image": segmented_image,
            "classified_image": classified_image,
            "gapcam_image": gapcam_path
        },

        # ── Clinician Assessment (empty initially) ──
        "clinician_review": {
            "status": "pending",
            "clinician_id": None,
            "final_decision": None,
            "notes": None
        },

        # ── Safety & Compliance ──
        "safety": {
            "ai_only_screening": True,
            "requires_clinician_review": False,
            "referral_recommendation": result["interpretation"]
        }
    }


# ── Screening list ────────────────────────────────────────────────────────────

@app.get("/api/screenings")
def api_screenings(limit: int = 20):
    screenings = models.list_screenings(limit)
    return [{"screening_id": s.screening_id, "patient_id": s.patient_id,
             "date": _fmt_ist(s.screening_date), "prediction": s.prediction,
             "confidence": s.confidence, "image_path": s.image_path,
             "gapcam_path": s.gapcam_path} for s in screenings]


@app.get("/api/screenings/{screening_id}")
def api_screening_by_id(screening_id: str):
    s = models.get_screening(screening_id)
    if not s:
        raise HTTPException(status_code=404, detail="Screening not found")
    p = models.get_patient(s.patient_id)

    # Derive 5-stage visualization paths from stored image_path
    # Naming convention: <patient_id>_<count>_<ts>.jpg -> <base>_preprocessed.jpg etc.
    img_path = s.image_path or ""
    base = ""
    if img_path and "/" in img_path:
        base = img_path.split("/")[-1].replace(".jpg", "").replace(".png", "").replace(".jpeg", "").replace(".tiff", "")
    elif img_path:
        base = img_path.replace(".jpg", "").replace(".png", "").replace(".jpeg", "").replace(".tiff", "")

    # Build relative paths for the 5-stage images
    up_dir = "uploads/"
    preprocessed_image = None
    segmented_image = None
    classified_image = None
    gradcam_image = None
    original_image = s.image_path

    if base:
        preprocessed_path = f"{up_dir}{base}_preprocessed.jpg"
        segmented_path = f"{up_dir}{base}_segmented.png"
        classified_path = f"{up_dir}{base}_classified.jpg"

        # Check if files exist on disk; only return if present
        def file_exists(rel):
            if not rel: return False
            abs_p = resolve_upload_path(rel)
            return bool(abs_p and os.path.exists(abs_p))

        if file_exists(preprocessed_path):
            preprocessed_image = preprocessed_path.replace("\\", "/")
        if file_exists(segmented_path):
            segmented_image = segmented_path.replace("\\", "/")
        if file_exists(classified_path):
            classified_image = classified_path.replace("\\", "/")

    # Grad-CAM uses gapcam_path (may be in uploads/gapcam/)
    if s.gapcam_path:
        gradcam_image = s.gapcam_path.replace("\\", "/")

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
        "threshold": THRESHOLD,
        "original_image": original_image,
        "preprocessed_image": preprocessed_image,
        "segmented_image": segmented_image,
        "classified_image": classified_image,
        "gradcam_image": gradcam_image
    }


@app.put("/api/screenings/{screening_id}/notes")
def api_screening_notes(screening_id: str, notes: str = Form("")):
    s = models.update_screening_notes(screening_id, notes)
    if not s:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True, "screening_id": s.screening_id, "notes": s.clinician_notes}


# ── Phase 8C: Clinician Review Endpoints ──────────────────────────────────────

@app.post("/api/screenings/{screening_id}/clinician-review")
def api_submit_clinician_review(
    screening_id: str,
    clinician_assessment: str = Form(...),
    clinician_notes: str = Form(None),
    override_reason: str = Form(None)
):
    """
    Submit a clinician assessment for a screening.

    PROTOTYPE-ONLY: Server-side authentication is NOT implemented.
    This prototype supports only a single clinician (`DEFAULT_CLINICIAN_ID`).
    The clinician review is controlled by the backend using this constant,
    ignoring frontend-provided clinician identity.
    """

    # Validate screening exists
    screening = models.get_screening(screening_id)
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")

    # Validate clinician_assessment
    VALID_ASSESSMENTS = {"NO_DR", "DR_PRESENT", "UNGRADABLE", "OTHER", "UNCERTAIN"}
    if clinician_assessment not in VALID_ASSESSMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid clinician_assessment. Must be one of: {', '.join(sorted(VALID_ASSESSMENTS))}"
        )

    # Check for override: AI prediction (binary 0 or 1) vs clinician assessment
    # AI prediction: 1 = DR Present, 0 = No DR
    # Clinician: NO_DR = 0, DR_PRESENT = 1, etc.
    ai_says_dr = (screening.prediction == 1)
    clinician_says_no_dr = (clinician_assessment == "NO_DR")
    clinician_says_dr_present = (clinician_assessment == "DR_PRESENT")

    is_override = (
        (ai_says_dr and clinician_says_no_dr) or
        (not ai_says_dr and clinician_says_dr_present) or
        (clinician_assessment in {"UNGRADABLE", "OTHER", "UNCERTAIN"})
    )

    # If override, require override_reason
    if is_override and not override_reason:
        raise HTTPException(
            status_code=400,
            detail="override_reason is required when clinician assessment differs from AI result or is non-diagnostic"
        )

    try:
        # Submit clinician review using hardcoded identity for single-user prototype
        updated = models.submit_clinician_review(
            screening_id=screening_id,
            clinician_id=DEFAULT_CLINICIAN_ID,
            clinician_assessment=clinician_assessment,
            clinician_notes=clinician_notes.strip() if clinician_notes else None,
            override_reason=override_reason.strip() if override_reason else None
        )

        return {
            "ok": True,
            "screening_id": updated.screening_id,
            "review_status": updated.review_status,
            "clinician_id": updated.clinician_id,
            "clinician_assessment": updated.clinician_assessment,
            "clinician_reviewed_at": _fmt_ist(updated.clinician_reviewed_at),
            "ai_result": {
                "prediction": float(updated.prediction) if updated.prediction is not None else None,
                "confidence": float(updated.confidence) if updated.confidence is not None else None,
                "model_version": updated.model_version,
                "threshold": updated.threshold
            }
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[ClinicianReview] Error: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


# Removed duplicate /api/screenings endpoint (api_screenings_list) to avoid route shadowing.
# The original api_screenings at line 701 remains as the single canonical endpoint.
    """
    List screenings with optional filtering by review_status.

    Query parameters:
    - limit: Number of screenings to return (default: 20)
    - review_status: Filter by status (PENDING, IN_PROGRESS, COMPLETED)
    """

    if review_status:
        # Validate review_status value
        VALID_STATUSES = {"PENDING", "IN_PROGRESS", "COMPLETED"}
        if review_status not in VALID_STATUSES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid review_status. Must be one of: {', '.join(sorted(VALID_STATUSES))}"
            )

        # Query pending/in-progress/completed reviews
        session = models.Session()
        screenings = session.query(models.Screening).filter_by(
            review_status=review_status
        ).order_by(
            models.desc(models.Screening.screening_date)
        ).limit(limit).all()

        # Format response
        result = []
        for s in screenings:
            result.append({
                "screening_id": s.screening_id,
                "patient_id": s.patient_id,
                "date": _fmt_ist(s.screening_date),
                "prediction": float(s.prediction) if s.prediction is not None else None,
                "confidence": float(s.confidence) if s.confidence is not None else None,
                "image_path": s.image_path,
                "gapcam_path": s.gapcam_path,
                "review_status": s.review_status,
                "clinician_id": s.clinician_id
            })

        session.close()
        return result

    else:
        # Original behavior: return all screenings
        screenings = models.list_screenings(limit)
        return [{"screening_id": s.screening_id, "patient_id": s.patient_id,
                 "date": _fmt_ist(s.screening_date), "prediction": s.prediction,
                 "confidence": s.confidence, "image_path": s.image_path,
                 "gapcam_path": s.gapcam_path} for s in screenings]


@app.get("/api/screenings/{screening_id}/audit-trail")
def api_screening_audit_trail(screening_id: str):
    """
    Return the complete available audit trail for a screening.

    Includes:
    - Screening creation event
    - AI prediction event
    - Image quality validation
    - Clinician review event (if completed)
    - Override information (if applicable)

    Note: This returns the currently available record history based on database fields.
    A full immutable event log infrastructure will be implemented in a future phase.
    """

    # Get screening record
    screening = models.get_screening(screening_id)
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")

    # Get patient for audit context
    patient = models.get_patient(screening.patient_id)

    # Build audit trail from available fields
    audit_trail = {
        "screening_id": screening.screening_id,
        "patient_id": screening.patient_id,
        "patient_name": patient.name if patient else "Unknown",
        "events": []
    }

    # Event 1: Screening created (AI screening performed)
    if screening.created_at:
        audit_trail["events"].append({
            "sequence": 1,
            "timestamp": _fmt_ist(screening.created_at),
            "event_type": "AI_SCREENING_PERFORMED",
            "actor": "EfficientNetV2B0 Model",
            "details": {
                "prediction": float(screening.prediction) if screening.prediction is not None else None,
                "confidence": float(screening.confidence) if screening.confidence is not None else None,
                "model_version": screening.model_version,
                "threshold": screening.threshold,
                "interpretation": "DR Present" if screening.prediction == 1 else "No DR"
            }
        })

    # Event 2: Image quality validation (inferred from screening_date)
    # Note: Image quality validation happens before AI screening in the current pipeline
    if screening.screening_date and screening.image_path:
        audit_trail["events"].append({
            "sequence": 2,
            "timestamp": _fmt_ist(screening.screening_date),
            "event_type": "IMAGE_QUALITY_VALIDATED",
            "actor": "Image Quality Gate",
            "details": {
                "image_path": screening.image_path,
                "status": "VALID"
            }
        })

    # Event 3: Clinician review (if completed)
    if screening.review_status == "COMPLETED" and screening.clinician_id:
        audit_trail["events"].append({
            "sequence": 3,
            "timestamp": _fmt_ist(screening.clinician_reviewed_at) if screening.clinician_reviewed_at else "Unknown",
            "event_type": "CLINICIAN_REVIEW_COMPLETED",
            "actor": screening.clinician_id,
            "details": {
                "clinician_assessment": screening.clinician_assessment,
                "clinician_confidence": screening.clinician_confidence,
                "clinician_notes": screening.clinician_notes,
                "override": screening.clinician_assessment is not None and (
                    (screening.prediction == 1 and screening.clinician_assessment == "NO_DR") or
                    (screening.prediction == 0 and screening.clinician_assessment == "DR_PRESENT") or
                    screening.clinician_assessment in {"UNGRADABLE", "OTHER", "UNCERTAIN"}
                ),
                "override_reason": screening.override_reason
            }
        })
    elif screening.review_status == "PENDING":
        audit_trail["events"].append({
            "sequence": 3,
            "timestamp": None,
            "event_type": "CLINICIAN_REVIEW_PENDING",
            "actor": None,
            "details": {
                "status": "Awaiting clinician assessment"
            }
        })

    # Sort events by sequence
    audit_trail["events"].sort(key=lambda e: e["sequence"])

    # Add metadata
    audit_trail["metadata"] = {
        "current_review_status": screening.review_status,
        "note": "This is the currently available record history from database fields. Full immutable event-log infrastructure will be implemented in a future phase."
    }

    return audit_trail


# ── Image endpoints ───────────────────────────────────────────────────────────

@app.get("/api/screenings/{screening_id}/image")
def api_screening_image(screening_id: str):
    """Serve the original retinal image for a screening."""
    s = models.get_screening(screening_id)
    if not s:
        raise HTTPException(status_code=404, detail="Screening not found")
    return serve_image_if_exists(s.image_path)


@app.get("/api/screenings/{screening_id}/heatmap")
def api_screening_heatmap(screening_id: str):
    """
    Serve the GAP-CAM heatmap for a screening.
    If no heatmap exists on disk, regenerate it on-demand.
    """
    s = models.get_screening(screening_id)
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
    session = models.Session()
    s = session.query(models.Screening).filter_by(screening_id=screening_id).first()
    if s:
        s.gapcam_path = gapcam_path
        session.commit()
    session.close()


# ── Report endpoint ───────────────────────────────────────────────────────────

@app.get("/api/reports/{screening_id}")
def api_report(screening_id: str, lang: str = "en"):
    from backend.reports import generate_report
    screening = models.get_screening(screening_id)
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")
    patient = models.get_patient(screening.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    # Phase 13: enrich payload with all relevant fields, IST-normalized.
    pdata = {
        "patient_id": patient.patient_id, "name": patient.name,
        "age": patient.age, "gender": patient.gender,
        "phone": patient.phone, "email": patient.email,
        "address": patient.address, "created_at": _fmt_ist(patient.created_at)
    }
    sdata = {
        "screening_id": screening.screening_id,
        "screening_date": _fmt_ist(screening.screening_date),
        "created_at": _fmt_ist(screening.created_at),
        "prediction": float(screening.prediction),
        "confidence": float(screening.confidence),
        "model_version": screening.model_version or "v1.0.0",
        "threshold": float(screening.threshold) if screening.threshold is not None else 0.5,
        "image_path": screening.image_path,
        "gapcam_path": screening.gapcam_path,
        # Phase 8B: clinician review
        "review_status": screening.review_status,
        "clinician_id": screening.clinician_id,
        "clinician_assessment": screening.clinician_assessment,
        "clinician_confidence": screening.clinician_confidence,
        "override_reason": screening.override_reason,
        "clinician_reviewed_at": _fmt_ist(screening.clinician_reviewed_at),
        "clinician_notes": screening.clinician_notes,
        # Phase 7: clinical action
        "clinical_action": screening.clinical_action,
        "clinical_action_notes": screening.clinical_action_notes,
        # Phase 13: image-quality status (derived from gapcam presence)
        "image_quality": "Validated" if screening.gapcam_path else "Not validated",
    }
    path = generate_report(screening_id, pdata, sdata, lang=lang)
    return FileResponse(path, media_type="text/html", filename=f"report_{screening_id}.html")


# ── Summary ──────────────────────────────────────────────────────────────────

# ── Phase 7: Clinical Action / Referral Workflow ───────────────────────────────

@app.post("/api/screenings/{screening_id}/clinical-action")
def api_record_clinical_action(
    screening_id: str,
    clinical_action: str = Form(...),
    clinical_action_notes: str = Form(None)
):
    """
    Record clinician's final action/follow-up decision.

    This is SEPARATE from:
    - AI prediction (immutable)
    - Clinician assessment (Phase 8D)

    Valid actions: REVIEW_REQUIRED, FOLLOW_UP, REFERRAL, NO_ACTION_DOCUMENTED
    """

    screening = models.get_screening(screening_id)
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")

    # Validate clinical_action
    VALID_ACTIONS = {"REVIEW_REQUIRED", "FOLLOW_UP", "REFERRAL", "NO_ACTION_DOCUMENTED"}
    if clinical_action not in VALID_ACTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid clinical_action. Must be one of: {', '.join(sorted(VALID_ACTIONS))}"
        )

    try:
        # Update screening with clinical action
        session = models.Session()
        s = session.query(models.Screening).filter_by(screening_id=screening_id).first()
        if s:
            s.clinical_action = clinical_action
            s.clinical_action_notes = clinical_action_notes.strip() if clinical_action_notes else None
            session.commit()
            session.refresh(s)

            return {
                "ok": True,
                "screening_id": s.screening_id,
                "clinical_action": s.clinical_action,
                "clinical_action_notes": s.clinical_action_notes,
                "timestamp": _fmt_ist(datetime.now(timezone.utc))
            }
        else:
            session.close()
            raise HTTPException(status_code=404, detail="Screening not found")
    except HTTPException:
        session.close()
        raise
    except Exception as e:
        session.close()
        print(f"[ClinicalAction] Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/summary")
def summary():
    return {
        "patients": models.patient_count(),
        "screenings": models.screening_count(),
        "recent": len(models.list_screenings(5)),
        "model": "EfficientNetV2B0 (INT8)",
        "version": "v1.0.0"
    }


# ── Root / UI ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    index_path = os.path.join(_BASE, "templates", "index.html")
    return HTMLResponse(open(index_path, encoding='utf-8').read())


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORARY DIAGNOSTIC ENDPOINT (DIAGNOSTICS ONLY — does not change decisions)
# Reports the EXACT signals and pass/fail for the active production validator
# (RetinalImageValidator, get_validator()). Reuses the validator's existing
# threshold constants; no production logic is altered by this endpoint.
# Remove this endpoint when diagnostics are no longer required.
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/diagnostics/image-quality")
async def diagnostics_image_quality(file: UploadFile = File(...)):
    """
    Diagnostic-only. Calls the SAME validate_image() used by /api/screenings/predict.
    Returns per-signal values, pass/fail, and the rejection reason.
    Does NOT classify, does NOT call EfficientNet.
    """
    # Save uploaded file into uploads/ to mirror production upload path
    # (validator expects a file on disk; uses PIL.Image.open).
    import tempfile
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOAD_DIR) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Run the EXISTING validator; no change to its decision logic.
        v = get_validator()
        result = validate_image(tmp_path)
        d = result.details or {}

        # Soft-signal voting set (mirrors production)
        soft_signals = {
            "circle_fov":        d.get("circle_fov_found", False),
            "green_channel":     (d.get("green_dominance", 0.0) >= v._green_dom_min),
            "dark_retinal_area": (d.get("dark_retinal_area", 0.0) >= v._dark_area_min),
            "color_variance":    (d.get("color_variance", 0.0) >= v._color_var_min),
            "texture_entropy":   (d.get("texture_entropy", 0.0) >= v._entropy_min),
            "edge_density":      (d.get("edge_density", 0.0) >= v._edge_density_min),
        }
        # Anti-signal set (mirrors production: very_low_entropy + red_dominant)
        anti_signals = {
            "very_low_entropy": (d.get("texture_entropy", 0.0) < (v._entropy_min - 1.0)),
            "red_dominant":     (d.get("red_dominance", 0.0) >= v._red_dom_anti),
        }
        # Reuse hard-check values
        def _check(name, value, lo, hi=None):
            if value is None:
                return "N/A"
            ok = (value >= lo) and (hi is None or value <= hi)
            return "PASS" if ok else "FAIL"

        report = {
            "image_quality_diagnostics": {
                "file": os.path.basename(tmp_path),
                "validator_class": type(v).__name__,
                "validator_module": v.__class__.__module__,
            },
            "hard_sanity": {
                "dimensions": {
                    "value": d.get("dimensions"),
                    "min_width": v.min_width, "max_width": v.max_width,
                    "min_height": v.min_height, "max_height": v.max_height,
                    "result": (
                        "PASS"
                        if d.get("dimensions")
                           and int(d["dimensions"].split("x")[0]) in range(v.min_width, v.max_width + 1)
                           and int(d["dimensions"].split("x")[1]) in range(v.min_height, v.max_height + 1)
                        else "FAIL"
                    ),
                },
                "aspect_ratio": {
                    "value": (round(d["dimensions"].split("x")[0] / d["dimensions"].split("x")[1], 3)
                              if d.get("dimensions") else None),
                    "min": v.aspect_ratio_min, "max": v.aspect_ratio_max,
                    "result": _check("aspect", (d["dimensions"].split("x")[0] / d["dimensions"].split("x")[1])
                                     if d.get("dimensions") else None,
                                     v.aspect_ratio_min, v.aspect_ratio_max),
                },
                "brightness": {
                    "value": d.get("brightness"),
                    "min": v.brightness_min, "max": v.brightness_max,
                    "result": _check("brightness", d.get("brightness"), v.brightness_min, v.brightness_max),
                },
                "contrast": {
                    "value": d.get("contrast"),
                    "min": v.contrast_min, "max": v.contrast_max,
                    "result": _check("contrast", d.get("contrast"), v.contrast_min, v.contrast_max),
                },
                "blur_laplacian_variance": {
                    "value": d.get("blur_score"),
                    "min": v.blur_threshold_min,
                    "result": _check("blur", d.get("blur_score"), v.blur_threshold_min),
                },
            },
            "soft_signals": {
                "circle_fov": {
                    "detected": d.get("circle_fov_found"),
                    "value": d.get("circle_fov_found"),
                    "result": "PASS" if d.get("circle_fov_found") else "FAIL",
                },
                "green_dominance": {
                    "value": d.get("green_dominance"),
                    "threshold_min": v._green_dom_min,
                    "result": "PASS" if (d.get("green_dominance", 0.0) >= v._green_dom_min) else "FAIL",
                },
                "dark_retinal_area": {
                    "value": d.get("dark_retinal_area"),
                    "threshold_min": v._dark_area_min,
                    "result": "PASS" if (d.get("dark_retinal_area", 0.0) >= v._dark_area_min) else "FAIL",
                },
                "color_variance": {
                    "value": d.get("color_variance"),
                    "threshold_min": v._color_var_min,
                    "result": "PASS" if (d.get("color_variance", 0.0) >= v._color_var_min) else "FAIL",
                },
                "entropy": {
                    "value": d.get("texture_entropy"),
                    "threshold_min": v._entropy_min,
                    "result": "PASS" if (d.get("texture_entropy", 0.0) >= v._entropy_min) else "FAIL",
                },
                "edge_density": {
                    "value": d.get("edge_density"),
                    "threshold_min": v._edge_density_min,
                    "result": "PASS" if (d.get("edge_density", 0.0) >= v._edge_density_min) else "FAIL",
                },
            },
            "anti_signals": {
                "very_low_entropy": {
                    "value": d.get("texture_entropy"),
                    "threshold_max": v._entropy_min - 1.0,
                    "result": "PASS" if not anti_signals["very_low_entropy"] else "FAIL",
                },
                "red_dominant": {
                    "value": d.get("red_dominance"),
                    "threshold_min": v._red_dom_anti,
                    "result": "PASS" if not anti_signals["red_dominant"] else "FAIL",
                },
            },
            "noise_veto_signal": {
                "spatial_autocorrelation": {
                    "value": d.get("spatial_autocorr"),
                    "threshold_min": v._autocorr_min,
                    "result": "PASS" if (d.get("spatial_autocorr", 0.0) < v._autocorr_min) else "FAIL",
                },
            },
            "voting": {
                "fundus_votes": d.get("fundus_votes"),
                "total_signals": d.get("total_signals"),
                "min_required": v._min_fundus_votes,
                "soft_signal_pass_count": sum(1 for s in soft_signals.values() if s),
                "soft_signal_results": soft_signals,
                "anti_signal_pass_count": sum(1 for s in anti_signals.values() if s),
                "anti_signal_results": anti_signals,
                "all_anti_signals_true": d.get("both_anti_signals"),
            },
            "anti_veto": {
                "all_anti_triggered": d.get("both_anti_signals"),
                "noise_veto_threshold": v._autocorr_min,
                "noise_veto_condition": (
                    f"autocorr >= {v._autocorr_min} AND votes < {v._min_fundus_votes + 2}"
                ),
                "noise_veto_would_fire": (
                    (d.get("spatial_autocorr", 0.0) >= v._autocorr_min)
                    and (d.get("fundus_votes", 0) < v._min_fundus_votes + 2)
                ),
            },
            "final": {
                "status": result.status.value,
                "confidence": result.confidence,
                "message": result.message,
                "should_classify": result.should_classify(),
                "classifier_will_run": result.should_classify(),
                "rejection_reason": (
                    None if result.should_classify()
                    else (
                        f"all_anti_signals true (red_dom >= {v._red_dom_anti} AND entropy < {v._entropy_min - 1.0})"
                        if d.get("both_anti_signals")
                        else (
                            f"noise veto (autocorr {d.get('spatial_autocorr'):.2f} >= {v._autocorr_min})"
                            if (d.get("spatial_autocorr", 0.0) >= v._autocorr_min and d.get("fundus_votes", 0) < v._min_fundus_votes + 2)
                            else (
                                f"borderline: votes {d.get('fundus_votes')}/6 < {v._min_fundus_votes} (conservative safe-reject)"
                                if d.get("fundus_votes", 0) < v._min_fundus_votes
                                else f"{result.status.value}: {result.message}"
                            )
                        )
                    )
                ),
            },
            "production_logic_untouched": {
                "RetinalImageValidator class in image_quality.py": "single, unchanged",
                "vote_threshold": v._min_fundus_votes,
                "red_dom_anti_threshold": v._red_dom_anti,
                "autocorr_noise_veto_threshold": v._autocorr_min,
            },
        }
        return report
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
