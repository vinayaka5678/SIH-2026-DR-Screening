from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime, timezone

Base = declarative_base()

# Use absolute path — computed at import time relative to this file's location
import pathlib
_DB_DIR = pathlib.Path(__file__).resolve().parent.parent / "database"
_DB_DIR.mkdir(parents=True, exist_ok=True)
_DB_PATH = _DB_DIR / "dr_screening.db"
ENGINE = create_engine(f"sqlite:///{_DB_PATH}", connect_args={"check_same_thread": False})


class Clinician(Base):
    __tablename__ = "clinicians"
    clinician_id = Column(String(20), primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    role = Column(String(50), default="Clinician")
    password_hash = Column(String(200))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Patient(Base):
    __tablename__ = "patients"
    patient_id = Column(String(20), primary_key=True)
    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(20))
    phone = Column(String(20))
    email = Column(String(100))
    address = Column(String(200))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    screenings = relationship("Screening", back_populates="patient", cascade="all, delete-orphan")


class Screening(Base):
    __tablename__ = "screenings"
    screening_id = Column(String(20), primary_key=True)
    patient_id = Column(String(20), ForeignKey("patients.patient_id"), nullable=False)
    screening_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    image_path = Column(String(200))
    prediction = Column(Float)
    confidence = Column(Float)
    model_version = Column(String(50), default="v1.0.0")
    threshold = Column(Float, default=0.5)
    gapcam_path = Column(String(200))
    clinician_notes = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Phase 8B: Clinician Review Fields (SEPARATE from AI result)
    clinician_id = Column(String(20))
    clinician_assessment = Column(String(50))  # NO_DR, DR_PRESENT, UNGRADABLE, OTHER, UNCERTAIN
    clinician_confidence = Column(Float)  # Clinician's own confidence (0-1)
    override_reason = Column(Text)
    clinician_reviewed_at = Column(String(50))  # ISO timestamp
    review_status = Column(String(20), default="PENDING")  # PENDING, IN_PROGRESS, COMPLETED

    # Phase 7: Clinician Action/Follow-up (SEPARATE from both AI and assessment)
    clinical_action = Column(String(50))  # REVIEW_REQUIRED, FOLLOW_UP, REFERRAL, NO_ACTION_DOCUMENTED
    clinical_action_notes = Column(Text)  # Clinician's notes for the final action

    patient = relationship("Patient", back_populates="screenings")


Base.metadata.create_all(ENGINE)
Session = sessionmaker(bind=ENGINE)
