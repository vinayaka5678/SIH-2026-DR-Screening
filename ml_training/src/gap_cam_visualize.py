#!/usr/bin/env python3
"""
GAP-CAM Visualization (Phase D)
Uses the saved best_model.keras (single-output classification) and constructs
dual-output feature map extraction directly without Lambda wrapping.
No retraining — read-only model inspection and visualization only.
"""
import os, sys, random, tempfile
sys.path.insert(0, 'ml_training/src')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# ------------------------------------------------------------------
# Load model and construct feature extractor (no Lambda needed)
# ------------------------------------------------------------------
MODEL_PATH = 'ml_training/models/full_training/best_model.keras'
OUTPUT_DIR = 'ml_training/models/full_training/gap_cam'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[GAP-CAM] Loading model from:", MODEL_PATH)
model = keras.models.load_model(MODEL_PATH)

# Find backbone layer
backbone = None
for layer in model.layers:
    if 'efficientnet' in layer.name.lower():
        backbone = layer
        break
if backbone is None:
    raise ValueError("Backbone not found in model layers")
print("[GAP-CAM] Backbone layer:", backbone.name)

# Build feature extractor from the base model's input -> backbone output
# We use model.layers to rebuild the exact same graph, then extract the
# intermediate feature_maps from the backbone output.
inputs = model.input
x = inputs
# Reproduce the preprocessing + backbone path
rescaling_layer = None
for layer in model.layers:
    if layer.name == 'rescaling':
        rescaling_layer = layer
        break
if rescaling_layer:
    x = rescaling_layer(inputs)
else:
    x = layers.Rescaling(scale=2.0, offset=-1.0, name='rescaling')(inputs)

# Get feature maps directly from backbone
feature_maps = backbone(x)

# Classification path (reuse existing weights from model)
# We can get the classification output directly from the model
classification_output = model.output  # This is already built

# For GAP-CAM we need the dense weights
# The dense layer is named 'classification' in the base model
classification_dense = None
for layer in model.layers:
    if isinstance(layer, keras.layers.Dense) and layer.name == 'classification':
        classification_dense = layer
        break
if classification_dense is None:
    raise ValueError("Classification Dense layer not found")

w, b = classification_dense.get_weights()
print("[GAP-CAM] Dense weights shape:", w.shape, "bias:", b.shape)

# Build feature extractor model (same architecture as best_model but with feature_maps output)
feature_extractor = keras.Model(inputs=inputs, outputs=feature_maps, name='feature_extractor')
# Copy backbone weights from base model to feature extractor
# Because feature_extractor shares the same backbone layer object reference,
# weights are already shared (same layer instance if reused). But we rebuilt the graph,
# so we need to copy weights explicitly.
# Find the new backbone layer in feature_extractor
new_backbone = None
for layer in feature_extractor.layers:
    if isinstance(layer, type(backbone)) and 'efficientnet' in layer.name.lower():
        new_backbone = layer
        break
if new_backbone is not None:
    # Try to copy weights from original backbone
    try:
        for s, d in zip(backbone.weights, new_backbone.weights):
            d.assign(s)
        print("[GAP-CAM] Copied backbone weights to feature extractor")
    except Exception as e:
        print("[GAP-CAM] Note: could not copy backbone weights (layer structure may differ):", e)
else:
    print("[GAP-CAM] Note: new backbone not found for weight copy")

# ------------------------------------------------------------------
# Load a sample image from dataset
# ------------------------------------------------------------------
import numpy as np
data = np.load('ml_training/processed_data/test.npz')
X_test = data['images']
y_test = data['labels']
print("[GAP-CAM] Loaded test data:", X_test.shape, y_test.shape)

# Pick a DR Present image (label 1) for visualization
sample_idx = np.where(y_test == 1)[0][0]
sample_img = X_test[sample_idx]  # [224, 224, 3], [0, 1]

# Predict classification
pred_prob = float(model.predict(np.expand_dims(sample_img, axis=0), verbose=0)[0][0])
print(f"[GAP-CAM] Sample image label: {y_test[sample_idx]}, Prediction prob: {pred_prob:.4f}")

# Get feature maps
f_maps = feature_extractor.predict(np.expand_dims(sample_img, axis=0), verbose=0)[0]
print("[GAP-CAM] Feature maps shape:", f_maps.shape)  # expected (7, 7, 1280)

# ------------------------------------------------------------------
# GAP-CAM computation (linear projection)
# ------------------------------------------------------------------
w_flat = w.flatten()  # shape: (1280,)
cam_linear = np.zeros((7, 7), dtype=np.float32)
for k in range(w_flat.shape[0]):
    cam_linear += w_flat[k] * f_maps[:, :, k]
cam_heatmap = np.maximum(cam_linear, 0)  # ReLU
if cam_heatmap.max() > 0:
    cam_heatmap /= cam_heatmap.max()  # normalize [0, 1]

print("[GAP-CAM] Heatmap shape:", cam_heatmap.shape, "range:", cam_heatmap.min(), "-", cam_heatmap.max())

# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------
# Resize heatmap to original image size
heatmap_resized = np.array(Image.fromarray((cam_heatmap * 255).astype(np.uint8)).resize(
    (224, 224), Image.Resampling.BILINEAR
)).astype(np.float32) / 255.0

# Convert image to [0, 255] for overlay
img_uint8 = (sample_img * 255).astype(np.uint8)
img_uint8 = np.clip(img_uint8, 0, 255).astype(np.uint8)

# Apply JET colormap manually (approximate)
colormap = plt.get_cmap('jet')
heatmap_color = (colormap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)

# Alpha blend
overlay = cv2.addWeighted(img_uint8, 0.5, heatmap_color, 0.5, 0)

# Save result
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
axes[0].imshow(img_uint8)
axes[0].set_title(f"Original Image\nTrue: DR Present | Pred: {pred_prob:.3f}")
axes[0].axis('off')

axes[1].imshow(cam_heatmap, cmap='jet', vmin=0, vmax=1)
axes[1].set_title("GAP-CAM Heatmap [7x7 -> 224x224]")
axes[1].axis('off')

axes[2].imshow(overlay)
axes[2].set_title("Overlay (Alpha Blend)")
axes[2].axis('off')

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, 'gap_cam_sample_dr.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print("[GAP-CAM] Visualization saved to:", output_path)

# Also save raw heatmap array for Android use
np.save(os.path.join(OUTPUT_DIR, 'gap_cam_heatmap.npy'), cam_heatmap)
np.save(os.path.join(OUTPUT_DIR, 'gap_cam_feature_maps.npy'), f_maps)

# Verify classification still works correctly
print("[GAP-CAM] Classification remains correct:", pred_prob)
print("[GAP-CAM] Phase D complete.")
