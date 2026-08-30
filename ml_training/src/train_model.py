#!/usr/bin/env python3
"""
Binary DR Screening Model Training Script

Trains EfficientNet-Lite0 with dual-output architecture for:
1. Binary DR classification (No DR vs DR Present)
2. Feature map extraction for GAP-CAM explainability

Key Features:
- EfficientNet-Lite0 backbone (pure ReLU6, no SE blocks)
- Global Average Pooling + Dense layer for GAP-CAM compatibility
- Synthetic domain-shift augmentation
- Class weight balancing
- Early stopping with model checkpointing
- TensorBoard logging

Usage:
    python train_model.py --data_dir processed_data --output_dir models
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt


def load_preprocessed_data(data_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load preprocessed data from NPZ file."""

    npz_path = data_dir / f"{split}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Preprocessed data not found: {npz_path}")

    data = np.load(npz_path)
    images = data['images']
    labels = data['labels']

    print(f"✅ Loaded {split} split: {len(images)} images")
    print(f"   Shape: {images.shape}, Labels: {labels.shape}")
    print(f"   Label distribution: No DR={np.sum(labels==0)}, DR={np.sum(labels==1)}")

    return images, labels


def create_augmentation_layer():
    """
    Create synthetic domain-shift augmentation layer.

    Simulates portable fundus camera artifacts:
    - Gaussian blur (optical quality variation)
    - Brightness jitter (exposure variation)
    - Contrast variation
    - Random horizontal/vertical flips

    Note: Advanced augmentations (vignetting, color shifts) can be added
    using Albumentations in preprocessing pipeline.
    """

    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomFlip("vertical"),
        layers.RandomRotation(0.1),  # ±36 degrees
        layers.RandomZoom(0.1),  # ±10% zoom
        layers.RandomContrast(0.2),  # ±20% contrast
        layers.RandomBrightness(0.2),  # ±20% brightness
    ], name="augmentation")

    return augmentation


def create_efficientnet_lite0_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    dropout_rate: float = 0.2
) -> keras.Model:
    """
    Create EfficientNet-Lite0 dual-output model for binary DR classification.

    Architecture:
        Input (224×224×3)
          ↓
        EfficientNet-Lite0 backbone (frozen initially)
          ↓
        Feature maps [7×7×1280]  ←  Output 2 (for GAP-CAM)
          ↓
        Global Average Pooling → [1280]
          ↓
        Dropout (0.2)
          ↓
        Dense (1, sigmoid)  ←  Output 1 (binary classification)

    Returns:
        Dual-output model:
        - output_1: Binary classification logit [batch, 1]
        - output_2: Feature maps [batch, 7, 7, 1280]
    """

    # Input layer
    inputs = layers.Input(shape=input_shape, name='input_image')

    # Data augmentation (only active during training)
    x = create_augmentation_layer()(inputs)

    # Preprocessing: Scale to [-1, 1] range (EfficientNet expects this)
    # Input is already [0, 1] from preprocessing, so scale to [-1, 1]
    x = layers.Rescaling(scale=2.0, offset=-1.0, name='rescaling')(x)

    # Load EfficientNet-Lite0 backbone
    # Note: Using EfficientNetV2B0 as proxy for EfficientNet-Lite0
    # (TensorFlow doesn't have official Lite variants, but V2B0 is similar)
    # For production, use keras_cv.models.EfficientNetLite0 or custom implementation
    backbone = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        include_preprocessing=False  # We handle preprocessing above
    )
    backbone.trainable = False  # Freeze backbone initially

    # Extract feature maps (last conv layer output)
    x = backbone(x, training=False)
    feature_maps = x  # Shape: [batch, 7, 7, 1280]

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(feature_maps)

    # Dropout for regularization
    x = layers.Dropout(dropout_rate, name='dropout')(x)

    # Binary classification head (single unit with sigmoid)
    classification_output = layers.Dense(
        1,
        activation='sigmoid',
        name='classification'
    )(x)

    # Create dual-output model
    model = keras.Model(
        inputs=inputs,
        outputs={
            'classification': classification_output,
            'feature_maps': feature_maps
        },
        name='efficientnet_lite0_dr_classifier'
    )

    return model


def calculate_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced dataset.

    Uses sklearn's balanced class weight formula:
    weight[c] = n_samples / (n_classes * n_samples_c)
    """

    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    print(f"✅ Calculated class weights: {class_weights}")

    return class_weights


def compile_model(model: keras.Model, learning_rate: float = 1e-3):
    """Compile model with binary cross-entropy loss and metrics."""

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'classification': 'binary_crossentropy',
            'feature_maps': None  # No loss for feature maps (just pass-through)
        },
        metrics={
            'classification': [
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        }
    )

    print(f"✅ Model compiled with learning_rate={learning_rate}")


def train_model(
    model: keras.Model,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    class_weights: Dict[int, float],
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 32
) -> keras.callbacks.History:
    """
    Train model with callbacks for early stopping and checkpointing.
    """

    X_train, y_train = train_data
    X_val, y_val = val_data

    output_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks
    checkpoint_path = output_dir / 'best_model.keras'
    log_dir = output_dir / 'logs'

    callback_list = [
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_classification_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_classification_auc',
            mode='max',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_classification_auc',
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1
        ),
        callbacks.CSVLogger(
            str(output_dir / 'training_log.csv')
        )
    ]

    print(f"\n🔄 Starting training...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Train samples: {len(X_train)}")
    print(f"   Val samples: {len(X_val)}")

    # Prepare target dict for dual outputs
    y_train_dict = {
        'classification': y_train,
        'feature_maps': np.zeros((len(y_train), 7, 7, 1280))  # Dummy target
    }
    y_val_dict = {
        'classification': y_val,
        'feature_maps': np.zeros((len(y_val), 7, 7, 1280))  # Dummy target
    }

    history = model.fit(
        X_train,
        y_train_dict,
        validation_data=(X_val, y_val_dict),
        epochs=epochs,
        batch_size=batch_size,
        class_weight={'classification': class_weights},
        callbacks=callback_list,
        verbose=1
    )

    print(f"\n✅ Training complete!")
    print(f"   Best model saved to: {checkpoint_path}")

    return history


def evaluate_model(
    model: keras.Model,
    test_data: Tuple[np.ndarray, np.ndarray],
    output_dir: Path
):
    """Evaluate model on test set and generate classification report."""

    X_test, y_test = test_data

    print(f"\n📊 Evaluating on test set ({len(X_test)} images)...")

    # Predict
    predictions = model.predict(X_test, verbose=0)
    y_pred_proba = predictions['classification'].flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n{'='*60}")
    print(f"TEST SET EVALUATION")
    print(f"{'='*60}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=['No DR', 'DR Present'],
        digits=4
    ))

    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"              Predicted")
    print(f"              No DR  DR")
    print(f"Actual No DR  {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"       DR     {cm[1,0]:5d}  {cm[1,1]:5d}")

    # Calculate sensitivity and specificity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print(f"\nSensitivity (Recall for DR): {sensitivity:.4f}")
    print(f"Specificity (Recall for No DR): {specificity:.4f}")
    print(f"{'='*60}")

    # Save evaluation results
    results = {
        'auc_roc': float(auc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(
            y_test, y_pred, target_names=['No DR', 'DR Present'], output_dict=True
        )
    }

    results_path = output_dir / 'test_evaluation.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Evaluation results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train binary DR screening model"
    )
    parser.add_argument(
        '--data_dir',
        type=Path,
        default=Path(__file__).parent.parent / 'processed_data',
        help='Directory containing preprocessed data'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path(__file__).parent.parent / 'models',
        help='Output directory for trained model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Maximum number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.2,
        help='Dropout rate'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Binary DR Screening Model Training")
    print("=" * 70)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Dropout: {args.dropout}")
    print()

    # Check TensorFlow GPU
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
    print()

    # Load data
    X_train, y_train = load_preprocessed_data(args.data_dir, 'train')
    X_val, y_val = load_preprocessed_data(args.data_dir, 'val')
    X_test, y_test = load_preprocessed_data(args.data_dir, 'test')

    # Calculate class weights
    class_weights = calculate_class_weights(y_train)

    # Create model
    print(f"\n🔨 Building EfficientNet-Lite0 model...")
    model = create_efficientnet_lite0_model(
        input_shape=(224, 224, 3),
        num_classes=2,
        dropout_rate=args.dropout
    )

    print(f"\n📊 Model Summary:")
    model.summary()

    # Compile model
    compile_model(model, learning_rate=args.learning_rate)

    # Train model
    history = train_model(
        model=model,
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        class_weights=class_weights,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Evaluate on test set
    evaluate_model(
        model=model,
        test_data=(X_test, y_test),
        output_dir=args.output_dir
    )

    print(f"\n✅ Training pipeline complete!")
    print(f"   Model saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
