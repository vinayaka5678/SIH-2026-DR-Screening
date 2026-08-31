from .database import Session, Patient, Screening, Clinician
from sqlalchemy import desc
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import random
import string

def _generate_id(prefix: str) -> str:
    """Generate a human-readable sequential ID."""
    session = Session()
    existing = session.query(Patient if prefix == "P" else Screening).all()
    session.close()
    numbers = []
    for item in existing:
        try:
            numbers.append(int(item.screening_id.split("-")[1]) if prefix == "S" else int(item.patient_id.split("-")[1]))
        except (IndexError, ValueError):
            numbers.append(0)
    next_num = max(numbers) + 1 if numbers else 1
    return f"{prefix}-{next_num:06d}"

# ── Patient CRUD ──────────────────────────────────────────────────────────────

def create_patient(name: str, age: int, gender: str = None,
                   phone: str = None, email: str = None, address: str = None) -> Patient:
    patient_id = _generate_id("P")
    session = Session()
    patient = Patient(
        patient_id=patient_id, name=name, age=age, gender=gender,
        phone=phone, email=email, address=address
    )
    session.add(patient)
    session.commit()
    session.refresh(patient)
    session.close()
    return patient

def get_patient(patient_id: str) -> Patient | None:
    session = Session()
    patient = session.query(Patient).filter_by(patient_id=patient_id).first()
    session.close()
    return patient

def search_patients(query: str):
    session = Session()
    q = session.query(Patient).filter(
        (Patient.patient_id.ilike(f"%{query}%")) |
        (Patient.name.ilike(f"%{query}%"))
    ).order_by(desc(Patient.created_at)).all()
    session.close()
    return q

def list_patients(limit: int = 50):
    session = Session()
    patients = session.query(Patient).order_by(desc(Patient.created_at)).limit(limit).all()
    session.close()
    return patients

def update_patient(patient_id: str, **fields) -> Patient | None:
    session = Session()
    patient = session.query(Patient).filter_by(patient_id=patient_id).first()
    if not patient:
        session.close()
        return None
    for k, v in fields.items():
        if hasattr(patient, k) and v is not None:
            setattr(patient, k, v)
    patient.updated_at = datetime.now(timezone.utc)
    session.commit()
    session.refresh(patient)
    session.close()
    return patient

def delete_patient(patient_id: str) -> bool:
    session = Session()
    patient = session.query(Patient).filter_by(patient_id=patient_id).first()
    if not patient:
        session.close()
        return False
    session.delete(patient)
    session.commit()
    session.close()
    return True

def patient_count() -> int:
    session = Session()
    count = session.query(Patient).count()
    session.close()
    return count

# ── Clinician CRUD ───────────────────────────────────────────────────────────

def get_clinician(clinician_id: str) -> Clinician | None:
    session = Session()
    clinician = session.query(Clinician).filter_by(clinician_id=clinician_id).first()
    session.close()
    return clinician

def get_clinician_by_email(email: str) -> Clinician | None:
    session = Session()
    clinician = session.query(Clinician).filter_by(email=email).first()
    session.close()
    return clinician

def update_clinician(clinician_id: str, **fields) -> Clinician | None:
    session = Session()
    clinician = session.query(Clinician).filter_by(clinician_id=clinician_id).first()
    if not clinician:
        session.close()
        return None
    for k, v in fields.items():
        if hasattr(clinician, k) and v is not None:
            setattr(clinician, k, v)
    session.commit()
    session.refresh(clinician)
    session.close()
    return clinician

def clinician_profile_response(clinician: Clinician):
    return {
        "clinician_id": clinician.clinician_id,
        "name": clinician.name,
        "email": clinician.email,
        "role": clinician.role,
        "created_at": str(clinician.created_at)
    }

# ── Screening CRUD ────────────────────────────────────────────────────────────

def create_screening(patient_id: str, prediction: float, confidence: float,
                     image_path: str = None, gapcam_path: str = None,
                     clinician_notes: str = None) -> Screening | None:
    if not get_patient(patient_id):
        return None
    screening_id = _generate_id("S")
    session = Session()
    screening = Screening(
        screening_id=screening_id, patient_id=patient_id,
        prediction=prediction, confidence=confidence,
        image_path=image_path, gapcam_path=gapcam_path,
        clinician_notes=clinician_notes,
        screening_date=datetime.now(timezone.utc)
    )
    session.add(screening)
    session.commit()
    session.refresh(screening)
    session.close()
    return screening

def get_screening(screening_id: str) -> Screening | None:
    session = Session()
    s = session.query(Screening).filter_by(screening_id=screening_id).first()
    session.close()
    return s

def get_screenings_for_patient(patient_id: str):
    session = Session()
    screenings = session.query(Screening).filter_by(patient_id=patient_id).order_by(desc(Screening.screening_date)).all()
    session.close()
    return screenings

def list_screenings(limit: int = 20):
    session = Session()
    screenings = session.query(Screening).order_by(desc(Screening.screening_date)).limit(limit).all()
    session.close()
    return screenings

def screening_count() -> int:
    session = Session()
    count = session.query(Screening).count()
    session.close()
    return count

def update_screening_notes(screening_id: str, notes: str) -> Screening | None:
    session = Session()
    screening = session.query(Screening).filter_by(screening_id=screening_id).first()
    if not screening:
        session.close()
        return None
    screening.clinician_notes = notes
    session.commit()
    session.refresh(screening)
    session.close()
    return screening
