#!/usr/bin/env python3
"""
TFLite Model Export with INT8 Quantization and GAP-CAM Weight Extraction

This script:
1. Loads trained Keras model
2. Converts to TensorFlow Lite with INT8 quantization
3. Extracts dense layer weights for GAP-CAM computation
4. Validates quantization accuracy
5. Generates Android-ready deployment bundle

Usage:
    python export_tflite.py --model_path models/best_model.keras --output_dir android_model
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras


def load_representative_dataset(data_dir: Path, num_samples: int = 100):
    train_data = np.load(data_dir / 'train.npz')
    images = train_data['images']
    indices = np.random.choice(len(images), size=min(num_samples, len(images)), replace=False)
    calibration_images = images[indices]
    print(f"Loaded {len(calibration_images)} images for quantization calibration")

    def representative_dataset_gen():
        for img in calibration_images:
            yield [np.expand_dims(img, axis=0).astype(np.float32)]
    return representative_dataset_gen


def export_dual_output_tflite(
    model_path: Path,
    output_path: Path,
    representative_dataset_gen,
    quantize: bool = True
) -> int:
    """
    Export a dual-output model (classification + feature_maps as float32)
    for GAP-CAM inference. Feature maps are kept at float32 so Android
    can apply dense weights for heatmap calculation.
    """
    model = keras.models.load_model(str(model_path))

    # For GAP-CAM: we need the feature_maps output, not just classification
    # Create a model that outputs both classification and feature_maps
    inference_model = keras.Model(
        inputs=model.input,
        outputs=[
            model.get_layer('classification').output,
            model.get_layer('feature_maps').output,
        ],
        name='inference_dual_output'
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)

    if quantize:
        print(f"\nApplying INT8 quantization on classification output (feature_maps kept float32)...")
        # Note: When specifying dual outputs, TFLite quantization applies
        # differently. We use default optimization but keep feature map output as float32.
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        # Classification output quantized to uint8; feature maps stay float32
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.float32  # Mixed outputs handled by interpreter
    else:
        print(f"\nConverting dual-output model to TFLite (no quantization)...")

    tflite_model = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    file_size = output_path.stat().st_size
    print(f"Dual-output TFLite model saved to: {output_path}")
    print(f"File size: {file_size / (1024**2):.2f} MB")
    return file_size


def extract_dense_weights(model: keras.Model) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract dense layer weights for GAP-CAM computation on Android.

    For a model with architecture:
        Global Average Pooling → [1280]
        Dense (1 unit) → [1]

    Returns:
        weights: shape [1280, 1] - weights connecting GAP to classification
        bias: shape [1] - bias term
    """

    # Find the classification dense layer
    dense_layer = None
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense) and layer.name == 'classification':
            dense_layer = layer
            break

    if dense_layer is None:
        raise ValueError("Could not find 'classification' Dense layer in model")

    weights, bias = dense_layer.get_weights()

    print(f"OK Extracted dense layer weights:")
    print(f"   Weights shape: {weights.shape}")
    print(f"   Bias shape: {bias.shape}")

    return weights, bias


def convert_to_tflite(
    model: keras.Model,
    representative_dataset_gen,
    output_path: Path,
    quantize: bool = True
) -> int:
    """
    Convert Keras model to TensorFlow Lite format with optional INT8 quantization.

    Returns:
        File size in bytes
    """

    # Create a model with single classification output for Android deployment
    # (We'll handle feature maps separately if needed)
    inference_model = keras.Model(
        inputs=model.input,
        outputs=model.get_layer('classification').output,
        name='inference_model'
    )

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)

    if quantize:
        print(f"\nPROCESSING: Applying INT8 quantization...")

        # Enable INT8 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen

        # Enforce INT8 for all operations
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8  # Quantized input
        converter.inference_output_type = tf.uint8  # Quantized output

        print(f"   Input type: uint8")
        print(f"   Output type: uint8")
    else:
        print(f"\nPROCESSING: Converting to TFLite (no quantization)...")

    tflite_model = converter.convert()

    # Save TFLite model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    file_size = output_path.stat().st_size
    print(f"OK TFLite model saved to: {output_path}")
    print(f"   File size: {file_size / (1024**2):.2f} MB ({file_size / 1024:.1f} KB)")

    return file_size


def validate_tflite_model(
    tflite_path: Path,
    test_data_dir: Path,
    num_test_samples: int = 100
):
    """
    Validate TFLite model accuracy on test set.
    """

    # Load test data
    test_data = np.load(test_data_dir / 'test.npz')
    X_test = test_data['images']
    y_test = test_data['labels']

    # Select random test samples
    indices = np.random.choice(len(X_test), size=min(num_test_samples, len(X_test)), replace=False)
    X_test_sample = X_test[indices]
    y_test_sample = y_test[indices]

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"\nSTATS: TFLite Model Details:")
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Input dtype: {input_details[0]['dtype']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    print(f"   Output dtype: {output_details[0]['dtype']}")

    # Check if model uses quantization
    is_quantized = input_details[0]['dtype'] == np.uint8

    print(f"\nPROCESSING: Running inference on {len(X_test_sample)} test samples...")

    predictions = []
    for img in X_test_sample:
        # Prepare input
        input_data = np.expand_dims(img, axis=0).astype(np.float32)

        if is_quantized:
            # Quantize input
            input_scale = input_details[0]['quantization'][0]
            input_zero_point = input_details[0]['quantization'][1]
            input_data = (input_data / input_scale + input_zero_point).astype(np.uint8)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        if is_quantized:
            # Dequantize output
            output_scale = output_details[0]['quantization'][0]
            output_zero_point = output_details[0]['quantization'][1]
            output = (output.astype(np.float32) - output_zero_point) * output_scale

        predictions.append(output[0, 0])

    predictions = np.array(predictions)
    y_pred = (predictions > 0.5).astype(int)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test_sample)

    # Calculate AUC if possible
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test_sample, predictions)
        print(f"OK TFLite validation (n={len(X_test_sample)}):")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   AUC-ROC: {auc:.4f}")
    except:
        print(f"OK TFLite validation (n={len(X_test_sample)}):")
        print(f"   Accuracy: {accuracy:.4f}")


def export_gapcam_weights(
    weights: np.ndarray,
    bias: np.ndarray,
    output_path: Path
):
    """
    Export dense layer weights in JSON format for Android GAP-CAM computation.

    GAP-CAM formula on Android:
        heatmap[x, y] = ReLU(Σ(weights[k] * feature_maps[x, y, k]))
    """

    # Convert to list for JSON serialization
    weights_list = weights.flatten().tolist()
    bias_value = float(bias[0])

    gapcam_data = {
        'weights': weights_list,
        'bias': bias_value,
        'num_channels': len(weights_list),
        'description': 'Dense layer weights for GAP-CAM heatmap generation'
    }

    with open(output_path, 'w') as f:
        json.dump(gapcam_data, f, indent=2)

    print(f"OK GAP-CAM weights exported to: {output_path}")
    print(f"   Number of channels: {len(weights_list)}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")


def create_deployment_bundle(
    output_dir: Path,
    model_info: dict
):
    """
    Create deployment metadata for Android integration.
    """

    deployment_info = {
        'model_version': '1.0.0',
        'model_architecture': 'EfficientNet-Lite0',
        'input_shape': [224, 224, 3],
        'output_classes': 2,
        'class_names': ['No DR', 'DR Present'],
        'normalization': 'Input images should be normalized to [0, 1] range',
        'files': {
            'tflite_model': 'dr_model_int8.tflite',
            'gapcam_weights': 'dense_weights.json'
        },
        **model_info
    }

    metadata_path = output_dir / 'deployment_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(deployment_info, f, indent=2)

    print(f"OK Deployment metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export trained model to TFLite with INT8 quantization"
    )
    parser.add_argument(
        '--model_path',
        type=Path,
        default=Path(__file__).parent.parent / 'models' / 'best_model.keras',
        help='Path to trained Keras model'
    )
    parser.add_argument(
        '--data_dir',
        type=Path,
        default=Path(__file__).parent.parent / 'processed_data',
        help='Directory containing preprocessed data for calibration'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path(__file__).parent.parent / 'android_model',
        help='Output directory for TFLite model and weights'
    )
    parser.add_argument(
        '--quantize',
        action='store_true',
        default=True,
        help='Apply INT8 quantization'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        default=True,
        help='Validate TFLite model on test set'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("TFLite Model Export with INT8 Quantization")
    print("=" * 70)
    print(f"Model path: {args.model_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Quantization: {'Enabled' if args.quantize else 'Disabled'}")
    print()

    # Load dual-output trained model
    # The dual-output model is saved as dual_output_model.keras
    dual_model_path = args.model_path
    if not dual_model_path.exists():
        dual_model_path = args.output_dir.parent / 'models' / 'dual_output_model.keras'

    print(f"Loading dual-output model from: {dual_model_path}")
    model = keras.models.load_model(str(dual_model_path))
    print(f"Model loaded successfully")
    print(f"Output names: {list(model.output_names)}")

    # Extract dense layer weights for GAP-CAM
    print(f"\nExtracting dense layer weights for GAP-CAM...")
    weights, bias = extract_dense_weights(model)

    # Load representative dataset for quantization
    if args.quantize:
        representative_dataset_gen = load_representative_dataset(args.data_dir)
    else:
        representative_dataset_gen = None

    # Export classification-only TFLite (INT8 quantized, smallest footprint)
    tflite_path = args.output_dir / 'dr_model_int8.tflite'
    file_size = convert_to_tflite(
        model=model,
        representative_dataset_gen=representative_dataset_gen,
        output_path=tflite_path,
        quantize=args.quantize
    )

    # Validate TFLite model
    if args.validate:
        validate_tflite_model(tflite_path, args.data_dir)

    # Export GAP-CAM weights
    gapcam_weights_path = args.output_dir / 'dense_weights.json'
    export_gapcam_weights(weights, bias, gapcam_weights_path)

    # Create deployment bundle
    model_info = {
        'tflite_size_mb': file_size / (1024**2),
        'quantized': args.quantize,
        'input_type': 'uint8' if args.quantize else 'float32',
        'output_type': 'uint8' if args.quantize else 'float32'
    }
    create_deployment_bundle(args.output_dir, model_info)

    print(f"\n{'='*70}")
    print(f"EXPORT COMPLETE")
    print(f"{'='*70}")
    print(f"TFLite model: {tflite_path}")
    print(f"GAP-CAM weights: {gapcam_weights_path}")
    print(f"Deployment metadata: {args.output_dir / 'deployment_metadata.json'}")
    print(f"\nAndroid Integration:")
    print(f"  1. Copy {tflite_path.name} to android/app/src/main/assets/")
    print(f"  2. Copy {gapcam_weights_path.name} to android/app/src/main/assets/")
    print(f"  3. Use TensorFlow Lite Interpreter to load model")
    print(f"  4. Use dense_weights.json for GAP-CAM heatmap generation")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
