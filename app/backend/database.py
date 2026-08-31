from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime, timezone

Base = declarative_base()

# Use absolute path — computed at import time relative to this file's location
_DB_PATH = "C:/Users/vinay/SIH-2026-DR-Screening/app/database/dr_screening.db"
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
    patient = relationship("Patient", back_populates="screenings")


Base.metadata.create_all(ENGINE)
Session = sessionmaker(bind=ENGINE)
