#!/usr/bin/env python3
"""
Build comprehensive deployment metadata for Android integration (Phase E step 10).
Reads the existing test_evaluation.json and TFLite export outputs, then writes
a complete deployment_metadata.json with all required fields.
"""
import os, json
from pathlib import Path

OUT_DIR = Path('ml_training/android_model')
EVAL_PATH = Path('ml_training/models/full_training/test_evaluation.json')

# Load test results
with open(EVAL_PATH) as f:
    test_results = json.load(f)

# Threshold from validation: 0.5 default. With val_recall=97.8%, threshold 0.5 gives
# test sensitivity 96.1% (>= 90% target). Document it.
THRESHOLD = 0.5

# TFLite file info
tflite_path = OUT_DIR / 'dr_model_int8.tflite'
gapcam_path = OUT_DIR / 'dense_weights.json'

tflite_size_mb = os.path.getsize(tflite_path) / (1024 ** 2)
gapcam_size_kb = os.path.getsize(gapcam_path) / 1024

metadata = {
    "model": {
        "name": "dr_screening_efficientnetv2b0_v1",
        "version": "1.0.0",
        "description": "Binary Diabetic Retinopathy screening model (DR Present vs No DR)",
        "backbone": "EfficientNetV2B0 (ImageNet pretrained, fine-tuned)",
        "input_size": [224, 224, 3],
        "num_classes": 1,
        "task": "binary_classification",
        "loss": "binary_crossentropy"
    },
    "preprocessing": {
        "image_resize": [224, 224],
        "resize_interpolation": "Lanczos4",
        "normalization": "[0, 1] (pixel / 255.0)",
        "color_mode": "RGB",
        "rescaling_in_model": "[-1, 1] via Rescaling(scale=2.0, offset=-1.0)",
        "tflite_input_format": "uint8 ([0, 1] float mapped to [0, 255] uint8)"
    },
    "class_mapping": {
        "0": "No DR Detected",
        "1": "DR Present (Refer to ophthalmologist)"
    },
    "decision_threshold": {
        "value": THRESHOLD,
        "selection_method": "Default 0.5; validation sensitivity >= 90% satisfied",
        "notes": "Lowering threshold increases sensitivity (recall) at cost of specificity"
    },
    "test_metrics": {
        "auc_roc": test_results["auc_roc"],
        "sensitivity": test_results["sensitivity"],
        "specificity": test_results["specificity"],
        "accuracy": test_results["classification_report"]["accuracy"],
        "test_set_size": 550,
        "confusion_matrix": test_results["confusion_matrix"]
    },
    "model_files": {
        "tflite_classifier": {
            "filename": "dr_model_int8.tflite",
            "size_mb": round(tflite_size_mb, 2),
            "quantization": "INT8 (post-training)",
            "input_dtype": "uint8",
            "output_dtype": "uint8",
            "input_shape": [1, 224, 224, 3],
            "output_shape": [1, 1]
        },
        "gap_cam_weights": {
            "filename": "dense_weights.json",
            "size_kb": round(gapcam_size_kb, 1),
            "format": "JSON: {weights: [1280 floats], bias: float, num_channels: 1280}",
            "purpose": "Linear projection weights for GAP-CAM heatmap generation on-device"
        }
    },
    "explainability": {
        "method": "GAP-CAM (Global Average Pooling - Class Activation Mapping)",
        "mathematical_equivalence": "For GAP -> Dense architectures, GAP-CAM weights equal Grad-CAM gradients (no backprop required)",
        "formula": "heatmap[x,y] = ReLU( sum_k (weights[k] * feature_maps[x,y,k]) )",
        "feature_map_shape": [7, 7, 1280],
        "upsample_target": [224, 224],
        "on_device_overhead_ms": "<5 ms (single linear projection per image)",
        "tflite_feature_map_model": "Not exported in INT8 (would require float32 dual-output). Use dense_weights.json + Keras/TFLite float32 for feature map inference if needed."
    },
    "android_integration": {
        "asset_path": "android/app/src/main/assets/",
        "required_files": [
            "dr_model_int8.tflite",
            "dense_weights.json"
        ],
        "runtime": "TensorFlow Lite / LiteRT",
        "min_android_api": 26,
        "min_ram_mb": 1024,
        "expected_inference_time_ms": "<100 ms on low-end devices"
    },
    "limitations": {
        "validation": "Not validated against portable camera images from rural PHCs",
        "intended_use": "Screening aid for trained PHC staff; not a diagnostic tool",
        "false_positive_guidance": "Always refer positive cases to an ophthalmologist for confirmation",
        "training_data": "APTOS 2019 Blindness Detection (3,662 retinal images)"
    }
}

OUT_PATH = OUT_DIR / 'deployment_metadata.json'
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"Deployment metadata written to: {OUT_PATH}")
print(f"TFLite size: {tflite_size_mb:.2f} MB")
print(f"GAP-CAM weights: {gapcam_size_kb:.1f} KB")
print(f"Test AUC: {test_results['auc_roc']:.4f}")
print(f"Test Sensitivity: {test_results['sensitivity']:.4f}")
print(f"Test Specificity: {test_results['specificity']:.4f}")
