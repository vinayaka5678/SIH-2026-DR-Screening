"""
Database Adapter Factory

Phase 9: Shared Data Architecture

Provides a unified interface for database operations.
Switches between Firebase (Firestore) and SQLite based on environment variable.

Usage:
    from database_adapter import get_models
    models = get_models()  # Returns appropriate adapter (firebase_adapter or models)
    screening = models.create_screening(...)
"""

import os
import logging

logger = logging.getLogger(__name__)


def get_database_backend() -> str:
    """
    Get configured database backend from environment.

    Returns:
        'firebase' or 'sqlite' (default: 'firebase')
    """
    backend = os.getenv('DATABASE_BACKEND', 'sqlite').lower()

    if backend not in ('firebase', 'sqlite'):
        logger.warning(f"Invalid DATABASE_BACKEND: {backend}, defaulting to firebase")
        return 'firebase'

    return backend


def get_models():
    """
    Get appropriate models module based on database backend.

    Returns:
        Module with CRUD functions (firebase_adapter or models)

    Examples:
        >>> models = get_models()
        >>> screening = models.create_screening(...)
        >>> patient = models.get_patient(patient_id)
    """
    backend = get_database_backend()

    if backend == 'firebase':
        try:
            from backend import firebase_adapter as models_module
            logger.info("Using Firebase backend (Firestore)")
            return models_module
        except ImportError as e:
            logger.error(f"Firebase adapter not available: {e}")
            raise
    else:  # sqlite
        try:
            from backend import models as models_module
            logger.info("Using SQLite backend")
            return models_module
        except ImportError as e:
            logger.error(f"SQLite models not available: {e}")
            raise


def is_firebase_enabled() -> bool:
    """
    Check if Firebase is enabled in configuration.

    Returns:
        bool: True if DATABASE_BACKEND is 'firebase'
    """
    return get_database_backend() == 'firebase'
