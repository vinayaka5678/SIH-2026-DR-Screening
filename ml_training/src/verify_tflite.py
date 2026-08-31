#!/usr/bin/env python3
"""
Phase F: TFLite Inference Verification
Compares TFLite predictions against Keras model on 100 test images.
Reports whether they are acceptably consistent.
"""
import os, sys, json
sys.path.insert(0, 'ml_training/src')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow import keras

MODEL_PATH = 'ml_training/models/full_training/best_model.keras'
TFLITE_PATH = 'ml_training/android_model/dr_model_int8.tflite'
TEST_DATA = 'ml_training/processed_data/test.npz'

# Load test data
data = np.load(TEST_DATA)
X_test = data['images']
y_test = data['labels']

# Load Keras model
keras_model = keras.models.load_model(MODEL_PATH)
keras_preds = keras_model.predict(X_test, verbose=0).flatten()

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_det = interpreter.get_input_details()[0]
output_det = interpreter.get_output_details()[0]

is_quantized = input_det['dtype'] == np.uint8
print("TFLite input dtype:", input_det['dtype'], "output:", output_det['dtype'])

tflite_preds = []
for i in range(len(X_test)):
    img = X_test[i]
    if is_quantized:
        scale, zero_pt = input_det['quantization']
        inp = np.expand_dims((img / scale + zero_pt).astype(np.uint8), axis=0)
    else:
        inp = np.expand_dims(img.astype(np.float32), axis=0)
    interpreter.set_tensor(input_det['index'], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_det['index'])
    if is_quantized:
        out_scale, out_zero = output_det['quantization']
        out = (out.astype(np.float32) - out_zero) * out_scale
    tflite_preds.append(out[0, 0])

tflite_preds = np.array(tflite_preds)

# Compare
differences = np.abs(keras_preds - tflite_preds)
max_diff = differences.max()
mean_diff = differences.mean()
std_diff = differences.std()

print(f"\n=== TFLite vs Keras Prediction Comparison ===")
print(f"N samples: {len(X_test)}")
print(f"Max absolute difference: {max_diff:.6f}")
print(f"Mean absolute difference: {std_diff:.6f}")
print(f"Median absolute difference: {np.median(differences):.6f}")

# Decision-level agreement
keras_binary = (keras_preds > 0.5).astype(int)
tflite_binary = (tflite_preds > 0.5).astype(int)
agreement = np.mean(keras_binary == tflite_binary)
print(f"\nDecision agreement (threshold 0.5): {agreement*100:.2f}%")

# AUC comparison
from sklearn.metrics import roc_auc_score
keras_auc = roc_auc_score(y_test, keras_preds)
tflite_auc = roc_auc_score(y_test, tflite_preds)
auc_diff = abs(keras_auc - tflite_auc)
print(f"Keras AUC: {keras_auc:.6f}")
print(f"TFLite AUC: {tflite_auc:.6f}")
print(f"AUC difference: {auc_diff:.6f}")

# Verdict
acceptable = auc_diff < 0.01 and agreement > 0.95
print(f"\nVerdict: {'PASS - TFLite predictions are acceptably consistent' if acceptable else 'FAIL - Check quantization settings'}")

# Save comparison results
results = {
    'n_samples': int(len(X_test)),
    'max_abs_diff': float(max_diff),
    'mean_abs_diff': float(mean_diff),
    'std_abs_diff': float(std_diff),
    'decision_agreement': float(agreement),
    'keras_auc': float(keras_auc),
    'tflite_auc': float(tflite_auc),
    'auc_difference': float(auc_diff),
    'verdict': 'PASS' if acceptable else 'FAIL'
}
with open('ml_training/android_model/tflite_verification.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to ml_training/android_model/tflite_verification.json")
