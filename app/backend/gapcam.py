"""
GAP-CAM heatmap generator for the DR screening backend.
Uses the dual_output_model.keras and dense_weights.json from the ML pipeline.
Reuses the same preprocessing as training (224x224, [0,1], Rescaling to [-1,1]).
"""
import os
import json
import numpy as np
from PIL import Image
import cv2

# Resolve paths relative to the project root (parent of app/), not CWD
_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../SIH-2026-DR-Screening/app
_PROJECT_ROOT = os.path.dirname(_APP_DIR)  # .../SIH-2026-DR-Screening
DUAL_MODEL_PATH = os.path.join(_PROJECT_ROOT, "ml_training", "models", "full_training", "dual_output_model.keras")
DENSE_WEIGHTS_PATH = os.path.join(_PROJECT_ROOT, "ml_training", "android_model", "dense_weights.json")
BACKBONE_NAME = "efficientnetv2-b0"  # confirmed from model inspection

# Load weights once
_weights = None
_bias = None
_feature_map_shape = None

def _load_weights():
    global _weights, _bias
    if _weights is None:
        with open(DENSE_WEIGHTS_PATH) as f:
            data = json.load(f)
        _weights = np.array(data["weights"], dtype=np.float32)
        _bias = float(data["bias"])
    return _weights, _bias


def generate_gapcam(image_path: str, output_path: str) -> str | None:
    """
    Generate a GAP-CAM heatmap overlay for the given image.
    Saves to output_path and returns the path on success, None on failure.
    """
    try:
        # Load dense weights
        weights, bias = _load_weights()

        # Load model (lazy, cached)
        import tensorflow as tf
        from tensorflow import keras
        model = keras.models.load_model(DUAL_MODEL_PATH, safe_mode=False)

        # Find backbone layer
        backbone = None
        for layer in model.layers:
            if "efficientnet" in layer.name.lower():
                backbone = layer
                break
        if backbone is None:
            print("[GAP-CAM] Backbone not found")
            return None

        # Build feature extractor: input -> rescaling -> backbone output
        inputs = model.input
        x = inputs
        for layer in model.layers:
            if layer.name == "rescaling":
                x = layer(x)
                break
        else:
            from tensorflow.keras import layers
            x = layers.Rescaling(scale=2.0, offset=-1.0, name="rescaling")(inputs)

        feature_maps = backbone(x)

        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        inp = np.expand_dims(arr, axis=0)

        # Extract feature maps
        extractor = keras.Model(inputs=inputs, outputs=feature_maps, name="feature_extractor")

        # Copy backbone weights from loaded model to extractor
        new_backbone = None
        for layer in extractor.layers:
            if "efficientnet" in layer.name.lower():
                new_backbone = layer
                break
        if new_backbone is not None and backbone is not None:
            for s, d in zip(backbone.weights, new_backbone.weights):
                d.assign(s)

        f_maps = extractor.predict(inp, verbose=0)[0]  # shape: (7, 7, 1280)
        print(f"[GAP-CAM] Feature maps: {f_maps.shape}")

        # Linear projection (GAP-CAM formula: ReLU(sum_k w_k * F_k))
        cam = np.zeros((7, 7), dtype=np.float32)
        for k in range(f_maps.shape[2]):
            cam += weights[k] * f_maps[:, :, k]
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam /= cam.max()

        # Resize heatmap to 224x224
        cam_img = Image.fromarray((cam * 255).astype(np.uint8), mode="L")
        cam_resized = np.array(cam_img.resize((224, 224), Image.Resampling.BILINEAR)).astype(np.float32) / 255.0

        # Apply JET colormap
        img_uint8 = np.clip((arr * 255), 0, 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Overlay: 50% original + 50% heatmap
        overlay = cv2.addWeighted(img_uint8, 0.5, heatmap_rgb, 0.5, 0)

        # Save overlay
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(overlay).save(output_path)
        print(f"[GAP-CAM] Saved: {output_path}")
        return output_path

    except Exception as e:
        print(f"[GAP-CAM] Error: {e}")
        import traceback; traceback.print_exc()
        return None
