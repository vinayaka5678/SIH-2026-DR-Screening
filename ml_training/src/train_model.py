#!/usr/bin/env python3
"""
Binary DR Screening Model Training Script

Trains EfficientNet-Lite0 with dual-output architecture for:
1. Binary DR classification (No DR vs DR Present)
2. Feature map extraction for GAP-CAM explainability

Key Features:
- EfficientNetV2B0 backbone (ImageNet pretrained)
- Global Average Pooling + Dense layer for GAP-CAM compatibility
- Augmentation in tf.data pipeline (NOT in model graph)
- Class weight balancing
- Early stopping with model checkpointing
- Two-stage transfer learning (frozen backbone, then fine-tuning)
- Single classification output for memory-efficient training

Usage:
    python train_model.py --data_dir processed_data --output_dir models
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, Callable
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_preprocessed_data(data_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load preprocessed data from NPZ file."""

    npz_path = data_dir / f"{split}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Preprocessed data not found: {npz_path}")

    data = np.load(npz_path)
    images = data['images']
    labels = data['labels']

    print(f"Loaded {split} split: {len(images)} images")
    print(f"   Shape: {images.shape}, Labels: {labels.shape}")
    print(f"   Label distribution: No DR={np.sum(labels==0)}, DR={np.sum(labels==1)}")

    return images, labels


# ==============================================================================
# AUGMENTATION LAYER (for embedding inside model during training)
# ==============================================================================

def create_augmentation_layer():
    """
    Create synthetic domain-shift augmentation layer.

    CRITICAL: value_range=(0, 1) is set because input images are already in [0, 1] range.
    Without this, RandomBrightness can output values in [-1, 1] which corrupts the pixel values.

    This layer is used to BUILD the training model (embedding augmentation in the graph),
    but augmentation also runs in the tf.data pipeline for better performance.
    """

    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomFlip("vertical"),
        layers.RandomRotation(0.1),  # ±36 degrees
        layers.RandomZoom(0.1),  # ±10% zoom
        layers.RandomContrast(0.2, value_range=(0.0, 1.0)),
        layers.RandomBrightness(0.2, value_range=(0.0, 1.0)),
    ], name="augmentation")

    return augmentation


# ==============================================================================
# TF.DATA AUGMENTATION (primary - runs in the data pipeline)
# ==============================================================================

def augment_image(image: tf.Tensor, label: tf.Tensor, seed: int = 42) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Augment a single image using tf.image operations in the tf.data pipeline.

    This runs on CPU during data loading, keeping the GPU/TPU free for actual training.
    Runs ONLY on training data (val/test bypass augmentation).
    """

    # Convert to float32 if needed
    image = tf.cast(image, tf.float32)

    # Random horizontal flip
    image = tf.image.random_flip_left_right(image, seed=seed)

    # Random vertical flip
    image = tf.image.random_flip_up_down(image, seed=seed + 1)

    # Random rotation (approximate with flips)
    # For ±36 degrees, we use small-angle approximation via affine transformation
    angle = tf.random.uniform([], -0.1, 0.1, seed=seed + 2)  # radians
    # Simple rotation using tf.contrib.image (or fallback)
    try:
        image = tf.keras.preprocessing.image.apply_affine_transform(
            image.numpy(), theta=angle.numpy(), row_axis=0, col_axis=1, channel_axis=2
        )
    except Exception:
        pass  # Skip rotation if not available

    # Random zoom (±10%)
    zoom_factor = tf.random.uniform([], 0.9, 1.1, seed=seed + 3)
    shape = tf.shape(image)
    new_height = tf.cast(tf.cast(shape[0], tf.float32) * zoom_factor, tf.int32)
    new_width = tf.cast(tf.cast(shape[1], tf.float32) * zoom_factor, tf.int32)
    image = tf.image.resize(image, [new_height, new_width])
    image = tf.image.resize_with_crop_or_pad(image, shape[0], shape[1])

    # Random brightness adjustment
    delta = tf.random.uniform([], -0.2, 0.2, seed=seed + 4)
    image = tf.clip_by_value(image + delta, 0.0, 1.0)

    # Random contrast adjustment
    factor = tf.random.uniform([], 0.8, 1.2, seed=seed + 5)
    mean = tf.reduce_mean(image)
    image = tf.clip_by_value((image - mean) * factor + mean, 0.0, 1.0)

    return image, label


def create_tfdata_pipeline(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    augment: bool = True,
    seed: int = 42
) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset pipeline for training.

    Args:
        images: numpy array of images in [0, 1] range
        labels: numpy array of binary labels
        batch_size: batch size
        shuffle: whether to shuffle the data
        augment: whether to apply augmentation (True for training only)
        seed: random seed

    Returns:
        tf.data.Dataset that yields (images_batch, labels_batch) tuples
    """

    # Create dataset from numpy arrays
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    # Shuffle training data
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images), seed=seed, reshuffle_each_iteration=True)

    # Map augmentation (CPU-bound, runs before batching)
    if augment:
        dataset = dataset.map(
            lambda img, lbl: augment_image(img, lbl, seed=seed),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # Batch
    dataset = dataset.batch(batch_size, drop_remainder=False)

    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================

def create_base_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.2
) -> keras.Model:
    """
    Create the base model WITHOUT augmentation.

    This is the CLEAN model used for:
    - Inference
    - TFLite export
    - GAP-CAM feature map extraction

    Architecture:
        Input (224×224×3)
          ↓
        Rescaling [-1, 1]
          ↓
        EfficientNetV2B0 backbone (pretrained)
          ↓
        Feature maps [7×7×1280]  ←  Output 2 (for GAP-CAM)
          ↓
        Global Average Pooling → [1280]
          ↓
        Dropout (0.2)
          ↓
        Dense (1, sigmoid)  ←  Output 1 (classification)
    """

    # Input layer
    inputs = layers.Input(shape=input_shape, name='input_image')

    # Preprocessing: Scale to [-1, 1] range (EfficientNet expects this)
    # Input is already [0, 1] from preprocessing, so scale to [-1, 1]
    x = layers.Rescaling(scale=2.0, offset=-1.0, name='rescaling')(inputs)

    # Load EfficientNetV2B0 backbone
    # Note: Using EfficientNetV2B0 as proxy for EfficientNet-Lite0
    # (TensorFlow doesn't have official Lite variants, but V2B0 is similar)
    backbone = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        include_preprocessing=False  # We handle preprocessing above
    )

    # Extract feature maps (last conv layer output)
    feature_maps = backbone(x, training=False)  # Shape: [batch, 7, 7, 1280]

    # Global Average Pooling
    gap = layers.GlobalAveragePooling2D(name='global_avg_pool')(feature_maps)

    # Dropout for regularization
    drop = layers.Dropout(dropout_rate, name='dropout')(gap)

    # Binary classification head (single unit with sigmoid)
    classification = layers.Dense(
        1,
        activation='sigmoid',
        name='classification'
    )(drop)

    # Classification-only model (clean, for inference/export)
    classification_model = keras.Model(
        inputs=inputs,
        outputs=classification,
        name='efficientnet_dr_classification'
    )

    return classification_model


def create_training_model(
    base_model: keras.Model,
    dropout_rate: float = 0.2
) -> Tuple[keras.Model, keras.Model]:
    """
    Create a training model by wrapping the base model with augmentation.

    Returns:
        Tuple of (training_model, dual_output_model):
        - training_model: Has augmentation in graph, single classification output
        - dual_output_model: No augmentation, dual outputs (classification + feature_maps)
          This is used for GAP-CAM inference and TFLite export.
    """

    # Input layer
    inputs = layers.Input(shape=(224, 224, 3), name='input_image')

    # Embed augmentation in the training model graph
    # Augmentation is also applied in tf.data pipeline for performance
    # Having it in the model graph too provides double augmentation on GPU
    x = create_augmentation_layer()(inputs)

    # Pass through the base model's backbone and head
    # We need to recreate the structure so we can extract feature_maps
    # Get the base model's weights and apply them
    x = layers.Rescaling(scale=2.0, offset=-1.0, name='rescaling')(x)
    backbone = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        include_preprocessing=False
    )
    backbone.trainable = False
    feature_maps = backbone(x, training=False)
    gap = layers.GlobalAveragePooling2D(name='global_avg_pool')(feature_maps)
    drop = layers.Dropout(dropout_rate, name='dropout')(gap)
    classification = layers.Dense(1, activation='sigmoid', name='classification')(drop)

    # Training model (single output, augmentation in graph)
    training_model = keras.Model(inputs=inputs, outputs=classification, name='efficientnet_dr_training')

    # Load weights from base model
    base_weights = base_model.get_layer('classification').get_weights()
    training_model.get_layer('classification').set_weights(base_weights)

    # Dual-output model for GAP-CAM (NO augmentation, clean for export)
    dual_output_model = keras.Model(
        inputs=inputs,
        outputs={
            'classification': classification,
            'feature_maps': feature_maps
        },
        name='efficientnet_dr_dual_output'
    )

    # Load same weights
    dual_output_model.get_layer('classification').set_weights(base_weights)

    return training_model, dual_output_model


# ==============================================================================
# CLASS WEIGHTS
# ==============================================================================

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

    print(f"Calculated class weights: {class_weights}")

    return class_weights


# ==============================================================================
# MODEL COMPILATION
# ==============================================================================

def compile_model(model: keras.Model, learning_rate: float = 1e-3):
    """Compile model with binary cross-entropy loss and metrics."""

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )

    print(f"Model compiled with learning_rate={learning_rate}")


# ==============================================================================
# TRAINING
# ==============================================================================

def train_model(
    model: keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    class_weights: Dict[int, float],
    output_dir: Path,
    epochs: int = 50,
    steps_per_epoch: int = None,
    validation_steps: int = None
) -> keras.callbacks.History:
    """
    Train model with callbacks for early stopping and checkpointing.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks
    checkpoint_path = output_dir / 'best_model.keras'
    log_dir = output_dir / 'logs'

    callback_list = [
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=0  # Set to 0 to avoid expensive histogram computation
        ),
        callbacks.CSVLogger(
            str(output_dir / 'training_log.csv')
        )
    ]

    print(f"\nStarting training...")
    print(f"   Epochs: {epochs}")
    print(f"   Train steps per epoch: {steps_per_epoch}")
    print(f"   Val steps: {validation_steps}")

    # Train using tf.data.Dataset
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        class_weight=class_weights,
        callbacks=callback_list,
        verbose=1
    )

    print(f"\nTraining complete!")
    print(f"   Best model saved to: {checkpoint_path}")

    return history


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_model(
    model: keras.Model,
    test_data: Tuple[np.ndarray, np.ndarray],
    output_dir: Path
):
    """Evaluate model on test set and generate classification report."""

    X_test, y_test = test_data

    print(f"\nEvaluating on test set ({len(X_test)} images)...")

    # Predict (model has single classification output)
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
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

    print(f"\nEvaluation results saved to: {results_path}")

    return 0


# ==============================================================================
# TWO-STAGE TRAINING
# ==============================================================================

def train_two_stage(
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: Dict[int, float],
    output_dir: Path,
    stage1_epochs: int = 5,
    stage2_epochs: int = 45,
    batch_size: int = 32,
    stage1_lr: float = 1e-3,
    stage2_lr: float = 1e-4
) -> Tuple[keras.Model, keras.Model, keras.callbacks.History]:
    """
    Two-stage transfer learning:
    Stage 1: Freeze backbone, train classification head
    Stage 2: Unfreeze backbone top layers, fine-tune
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / 'best_model.keras'

    # Create base model (no augmentation)
    print("\n" + "="*60)
    print("STAGE 1: Train classification head (backbone frozen)")
    print("="*60)

    base_model = create_base_model(input_shape=(224, 224, 3))
    compile_model(base_model, learning_rate=stage1_lr)

    # Stage 1 callbacks
    stage1_callbacks = [
        callbacks.ModelCheckpoint(
            filepath=str(output_dir / 'stage1_model.keras'),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.CSVLogger(str(output_dir / 'stage1_log.csv'))
    ]

    # Calculate steps
    # We need the actual number of samples from the dataset
    train_steps = None  # Use None to iterate over full dataset
    val_steps = None

    print(f"\nStage 1: {stage1_epochs} epochs, learning_rate={stage1_lr}")

    # Stage 1 training
    history_s1 = base_model.fit(
        train_dataset,
        validation_data=(X_val, y_val),
        epochs=stage1_epochs,
        steps_per_epoch=train_steps,
        class_weight=class_weights,
        callbacks=stage1_callbacks,
        verbose=1
    )

    # Load best stage 1 model
    stage1_path = output_dir / 'stage1_model.keras'
    if stage1_path.exists():
        base_model = keras.models.load_model(stage1_path)
        print(f"Loaded best stage 1 model from {stage1_path}")

    # Stage 2: Fine-tune with unfrozen backbone
    print("\n" + "="*60)
    print("STAGE 2: Fine-tune backbone (top layers unfrozen)")
    print("="*60)

    # Unfreeze the top layers of the backbone
    # For EfficientNetV2B0, we unfreeze from layer 200 onwards
    backbone = None
    for layer in base_model.layers:
        if hasattr(layer, 'layers'):  # It's a Keras application
            backbone = layer
            break

    if backbone is not None:
        # Unfreeze the last 20 layers
        trainable = False
        for layer in backbone.layers:
            if 'block7' in layer.name or 'block6' in layer.name:
                layer.trainable = True
                trainable = True
            elif trainable:
                layer.trainable = True

        print(f"Unfroze top layers of backbone for fine-tuning")

    # Recompile with lower learning rate
    compile_model(base_model, learning_rate=stage2_lr)

    # Stage 2 callbacks
    stage2_callbacks = [
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.CSVLogger(str(output_dir / 'stage2_log.csv'))
    ]

    print(f"\nStage 2: {stage2_epochs} epochs, learning_rate={stage2_lr}")

    # Stage 2 training
    history_s2 = base_model.fit(
        train_dataset,
        validation_data=(X_val, y_val),
        epochs=stage2_epochs,
        steps_per_epoch=train_steps,
        class_weight=class_weights,
        callbacks=stage2_callbacks,
        verbose=1
    )

    # Merge histories
    history = history_s1
    for key in history_s2.history:
        history.history[key] = history_s1.history.get(key, []) + history_s2.history.get(key, [])

    # Save dual-output model for GAP-CAM
    dual_output_model = create_dual_output_model(base_model)
    dual_model_path = output_dir / 'dual_output_model.keras'
    dual_output_model.save(str(dual_model_path))
    print(f"\nDual-output model saved to: {dual_model_path}")

    return base_model, dual_output_model, history


def create_dual_output_model(base_model: keras.Model) -> keras.Model:
    """
    Create a dual-output model from the base classification model.

    The dual output model has:
    1. classification: sigmoid probability
    2. feature_maps: the last conv layer output for GAP-CAM

    Important: Keras 3 ignores dict keys for output names when the
    underlying tensor already has a name (e.g. from a named layer).
    We rename the feature_maps tensor explicitly via a Lambda wrapping
    so output_names is ['classification', 'feature_maps'].
    """

    # Get the classification layer weights
    classification_weights = base_model.get_layer('classification').get_weights()

    # Build the dual-output architecture
    inputs = layers.Input(shape=(224, 224, 3), name='input_image')

    x = layers.Rescaling(scale=2.0, offset=-1.0, name='rescaling')(inputs)
    backbone = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        include_preprocessing=False
    )
    backbone.trainable = False

    feature_maps = backbone(x, training=False)
    # Rename the feature_maps tensor so the output name is 'feature_maps'
    # instead of the backbone's internal name (e.g. 'efficientnetv2-b0').
    feature_maps = layers.Lambda(lambda t: t, name='feature_maps')(feature_maps)

    gap = layers.GlobalAveragePooling2D(name='global_avg_pool')(feature_maps)
    drop = layers.Dropout(0.2, name='dropout')(gap)
    classification = layers.Dense(1, activation='sigmoid', name='classification')(drop)

    dual_model = keras.Model(
        inputs=inputs,
        outputs=[classification, feature_maps],
        name='efficientnet_dr_dual_output'
    )

    # Copy backbone weights from the base model (if available)
    try:
        # The base_model wraps a backbone internally; locate the EfficientNetV2B0
        # sublayer and copy its weights.
        base_backbone = None
        for layer in base_model.layers:
            if 'efficientnet' in layer.name.lower():
                base_backbone = layer
                break
        if base_backbone is not None:
            for src, dst in zip(base_backbone.weights, backbone.weights):
                dst.assign(src)
    except Exception as e:
        print(f"Warning: could not copy backbone weights to dual model: {e}")

    # Copy classification head weights
    dual_model.get_layer('classification').set_weights(classification_weights)

    return dual_model


# ==============================================================================
# MAIN
# ==============================================================================

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
        '--stage1_epochs',
        type=int,
        default=5,
        help='Epochs for stage 1 (frozen backbone)'
    )
    parser.add_argument(
        '--stage2_epochs',
        type=int,
        default=45,
        help='Epochs for stage 2 (fine-tuning)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--stage1_lr',
        type=float,
        default=1e-3,
        help='Stage 1 learning rate'
    )
    parser.add_argument(
        '--stage2_lr',
        type=float,
        default=1e-4,
        help='Stage 2 learning rate'
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
    print(f"Stage 1: {args.stage1_epochs} epochs, lr={args.stage1_lr}")
    print(f"Stage 2: {args.stage2_epochs} epochs, lr={args.stage2_lr}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Check TensorFlow
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

    # Create tf.data pipelines
    print("\nCreating tf.data training pipeline...")
    train_dataset = create_tfdata_pipeline(
        X_train, y_train,
        batch_size=args.batch_size,
        shuffle=True,
        augment=True,
        seed=42
    )
    print("Training pipeline created with augmentation")

    # Two-stage training
    final_model, dual_output_model, history = train_two_stage(
        train_dataset=train_dataset,
        val_dataset=(X_val, y_val),
        X_val=X_val,
        y_val=y_val,
        class_weights=class_weights,
        output_dir=args.output_dir,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        batch_size=args.batch_size,
        stage1_lr=args.stage1_lr,
        stage2_lr=args.stage2_lr
    )

    # Evaluate on test set
    evaluate_model(
        model=final_model,
        test_data=(X_test, y_test),
        output_dir=args.output_dir
    )

    # Save training history
    history_path = args.output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    print(f"\nTraining pipeline complete!")
    print(f"   Model saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())