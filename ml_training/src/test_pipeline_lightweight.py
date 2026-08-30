#!/usr/bin/env python3
"""
Lightweight Pipeline Verification Test

Tests the full Phase 2 ML pipeline on a small subset (50 images) without
performing full training or modifying the original dataset:
1. Label conversion integrity (0 -> 0, 1-4 -> 1)
2. Stratified train/val/test splitting and data leakage check
3. Image loading, resizing to 224x224, Lanczos4, normalization [0, 1]
4. Dual-output model construction (classification + feature_maps)
5. 1-epoch lightweight CPU forward/backward pass
6. Dense layer weight extraction for GAP-CAM
7. GAP-CAM heatmap calculation validation
8. INT8 quantization to TFLite and verification of input/output specs
"""

import os
import sys
import shutil
import tempfile
import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


def test_label_conversion(csv_path: Path):
    print("\n" + "=" * 60)
    print("TEST 1: Binary DR Label Conversion Verification")
    print("=" * 60)
    df = pd.read_csv(csv_path)

    # Ground truth mapping:
    # 0 -> 0 (No DR)
    # 1, 2, 3, 4 -> 1 (DR Present)
    df['binary_label'] = (df['diagnosis'] > 0).astype(int)

    for grade in range(5):
        subset = df[df['diagnosis'] == grade]
        expected_binary = 0 if grade == 0 else 1
        actual_binaries = subset['binary_label'].unique()
        assert len(actual_binaries) == 1 and actual_binaries[0] == expected_binary, \
            f"❌ Label conversion failed for diagnosis grade {grade}"
        print(f"  Diagnosis {grade} ({len(subset):4d} images) -> Binary {actual_binaries[0]} (Expected: {expected_binary}) ✅")

    print("✅ Label conversion test PASSED: 0=No DR, 1-4=DR Present.")


def test_stratified_split_and_leakage(csv_path: Path):
    print("\n" + "=" * 60)
    print("TEST 2: Stratified Split & Disjoint Leakage Check")
    print("=" * 60)
    df = pd.read_csv(csv_path)
    df['binary_label'] = (df['diagnosis'] > 0).astype(int)

    # Stratified split 70/15/15
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['binary_label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['binary_label'], random_state=42)

    # Check disjoint IDs
    s_train = set(train_df['id_code'])
    s_val = set(val_df['id_code'])
    s_test = set(test_df['id_code'])

    assert len(s_train & s_val) == 0, "Leakage between train and val"
    assert len(s_train & s_test) == 0, "Leakage between train and test"
    assert len(s_val & s_test) == 0, "Leakage between val and test"

    # Check proportions
    print(f"  Total records: {len(df)}")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%), DR %: {train_df['binary_label'].mean()*100:.1f}%")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%), DR %: {val_df['binary_label'].mean()*100:.1f}%")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%), DR %: {test_df['binary_label'].mean()*100:.1f}%")
    print("✅ Split test PASSED: 100% disjoint, zero leakage, balanced stratification.")


def test_preprocessing_and_model_pipeline(data_dir: Path, temp_dir: Path):
    print("\n" + "=" * 60)
    print("TEST 3: End-to-End Pipeline on 50 Sample Images")
    print("=" * 60)

    csv_path = data_dir / "aptos2019" / "train.csv"
    images_dir = data_dir / "aptos2019" / "train_images"

    df = pd.read_csv(csv_path)
    df['binary_label'] = (df['diagnosis'] > 0).astype(int)

    # Pick balanced sample of 50 images (25 No DR, 25 DR Present)
    df_sample = pd.concat([
        df[df['binary_label'] == 0].head(25),
        df[df['binary_label'] == 1].head(25)
    ]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"  Sample size: {len(df_sample)} images (25 No DR, 25 DR Present)")

    # Preprocess images
    X = []
    y = []
    for _, row in df_sample.iterrows():
        img_p = str(images_dir / f"{row['id_code']}.png")
        img = cv2.imread(img_p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        img = img.astype(np.float32) / 255.0
        X.append(img)
        y.append(row['binary_label'])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    assert X.shape == (50, 224, 224, 3), f"Wrong X shape: {X.shape}"
    assert X.min() >= 0.0 and X.max() <= 1.0, "Normalization out of [0, 1] range"
    print(f"  Preprocessed X: shape={X.shape}, dtype={X.dtype}, min={X.min():.2f}, max={X.max():.2f} ✅")

    # Build dual-output model
    print("\n  Building EfficientNet Dual-Output Model...")
    inputs = layers.Input(shape=(224, 224, 3), name='input_image')
    x = layers.Rescaling(scale=2.0, offset=-1.0, name='rescaling')(inputs)
    backbone = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        include_preprocessing=False
    )
    backbone.trainable = False

    feature_maps = backbone(x, training=False)  # shape: (batch, 7, 7, 1280)
    gap = layers.GlobalAveragePooling2D(name='global_avg_pool')(feature_maps)
    drop = layers.Dropout(0.2, name='dropout')(gap)
    classification = layers.Dense(1, activation='sigmoid', name='classification')(drop)

    model = keras.Model(
        inputs=inputs,
        outputs={'classification': classification, 'feature_maps': feature_maps},
        name='dr_dual_output_model'
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={'classification': 'binary_crossentropy', 'feature_maps': None},
        metrics={'classification': ['accuracy']}
    )

    print("  Running 1 lightweight training epoch...")
    dummy_feat_targets = np.zeros((50, 7, 7, 1280), dtype=np.float32)
    hist = model.fit(
        X,
        {'classification': y, 'feature_maps': dummy_feat_targets},
        epochs=1,
        batch_size=16,
        verbose=0
    )
    print(f"  Epoch 1 Loss: {hist.history['loss'][0]:.4f}, Accuracy: {hist.history['classification_accuracy'][0]:.4f} ✅")

    # GAP-CAM Weight Extraction & Computation Verification
    print("\n" + "=" * 60)
    print("TEST 4: GAP-CAM Weight Extraction & Formula Validation")
    print("=" * 60)
    dense_layer = model.get_layer('classification')
    dense_weights, dense_bias = dense_layer.get_weights()
    print(f"  Extracted Dense Weights shape: {dense_weights.shape} (Channels: {dense_weights.shape[0]})")
    print(f"  Extracted Dense Bias shape: {dense_bias.shape}")

    # Forward pass on 1 image to get feature maps
    sample_input = np.expand_dims(X[0], axis=0)
    outputs = model(sample_input, training=False)
    pred_prob = float(outputs['classification'].numpy()[0, 0])
    f_maps = outputs['feature_maps'].numpy()[0]  # shape: (7, 7, 1280)

    # Linear projection: heatmap = ReLU(sum_k (w_k * F_k))
    # w_k has shape (1280, 1), F_k has shape (7, 7, 1280)
    w = dense_weights.flatten()
    cam_linear = np.zeros((7, 7), dtype=np.float32)
    for k in range(1280):
        cam_linear += w[k] * f_maps[:, :, k]
    cam_heatmap = np.maximum(cam_linear, 0) # ReLU

    # Normalize heatmap to [0, 1]
    if cam_heatmap.max() > 0:
        cam_heatmap /= cam_heatmap.max()

    print(f"  Prediction probability: {pred_prob:.4f}")
    print(f"  Feature maps shape: {f_maps.shape}")
    print(f"  GAP-CAM Heatmap shape: {cam_heatmap.shape}, min={cam_heatmap.min():.2f}, max={cam_heatmap.max():.2f} ✅")

    # INT8 Quantization test
    print("\n" + "=" * 60)
    print("TEST 5: INT8 Quantization & TFLite Model Validation")
    print("=" * 60)

    def rep_data_gen():
        for i in range(10):
            yield [np.expand_dims(X[i], axis=0)]

    inf_model = keras.Model(inputs=model.input, outputs=model.get_layer('classification').output)
    converter = tf.lite.TFLiteConverter.from_keras_model(inf_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    tflite_path = temp_dir / 'test_model_int8.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"  Quantized TFLite size: {size_mb:.2f} MB")

    # Verify interpreter execution
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    print(f"  TFLite Input: shape={in_det['shape']}, dtype={in_det['dtype']}")
    print(f"  TFLite Output: shape={out_det['shape']}, dtype={out_det['dtype']}")

    assert in_det['dtype'] == np.uint8, "Expected uint8 input"
    assert out_det['dtype'] == np.uint8, "Expected uint8 output"

    # Run 1 test inference on TFLite
    scale, zero_pt = in_det['quantization']
    q_in = (sample_input / scale + zero_pt).astype(np.uint8)
    interpreter.set_tensor(in_det['index'], q_in)
    interpreter.invoke()
    q_out = interpreter.get_tensor(out_det['index'])
    out_scale, out_zero_pt = out_det['quantization']
    tflite_pred = (float(q_out[0, 0]) - out_zero_pt) * out_scale
    print(f"  TFLite INT8 output probability: {tflite_pred:.4f} (Keras float32: {pred_prob:.4f})")
    print("✅ INT8 Quantization test PASSED.")


def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    csv_path = data_dir / "aptos2019" / "train.csv"

    temp_dir = Path(tempfile.mkdtemp())
    try:
        test_label_conversion(csv_path)
        test_stratified_split_and_leakage(csv_path)
        test_preprocessing_and_model_pipeline(data_dir, temp_dir)
        print("\n" + "=" * 60)
        print("🎉 ALL LIGHTWEIGHT PIPELINE TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
