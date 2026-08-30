# ML Training Pipeline
**Diabetic Retinopathy Binary Classification Model**

---

## Overview

Python-based training pipeline for EfficientNet-Lite0 binary DR classification model with dual-output architecture (classification + feature maps for GAP-CAM).

---

## Directory Structure

```
ml_training/
├── notebooks/              # Jupyter notebooks for experimentation
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_grad_cam_validation.ipynb
├── src/                    # Python training scripts
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── export_tflite.py
│   └── evaluate.py
├── data/                   # Dataset storage (not tracked by Git)
│   ├── aptos2019/
│   ├── eyepacs/
│   └── idrid/
├── models/                 # Trained models (not tracked by Git)
│   ├── checkpoints/
│   └── final/
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create Python Virtual Environment
```bash
cd ml_training
python -m venv venv_ml
.\venv_ml\Scripts\activate  # Windows
source venv_ml/bin/activate  # Linux/macOS
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Datasets

### Required Datasets (Download Separately)

1. **APTOS 2019 Blindness Detection**
   - Source: [Kaggle APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection)
   - Size: ~5,590 images
   - Population: Indian (Tamil Nadu)
   - Place in: `data/aptos2019/`

2. **EyePACS / Kaggle DR Detection 2015**
   - Source: [Kaggle Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection)
   - Size: ~88,000 images
   - Population: Mixed US
   - Place in: `data/eyepacs/`

3. **IDRiD (Optional - Small Dataset)**
   - Source: [ISBI 2018 IDRiD Challenge](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)
   - Size: 516 images
   - Population: Indian (Maharashtra)
   - Place in: `data/idrid/`

---

## Training Pipeline

### Phase 1: Data Preprocessing
```bash
python src/data_preprocessing.py \
  --aptos_dir data/aptos2019 \
  --eyepacs_dir data/eyepacs \
  --output_dir data/processed \
  --binary_threshold 0  # 0 = No DR, 1-4 = DR Present
```

### Phase 2: Model Training
```bash
python src/train_model.py \
  --data_dir data/processed \
  --model efficientnet-lite0 \
  --input_size 224 \
  --batch_size 32 \
  --epochs 20 \
  --output_dir models/checkpoints
```

### Phase 3: Model Export & Quantization
```bash
python src/export_tflite.py \
  --keras_model models/checkpoints/best_model.keras \
  --output_dir models/final \
  --quantize int8
```

Outputs:
- `models/final/dr_model_int8.tflite` (~5.5 MB)
- `models/final/dense_weights.json` (~5 KB)

### Phase 4: Evaluation
```bash
python src/evaluate.py \
  --tflite_model models/final/dr_model_int8.tflite \
  --test_data data/processed/test \
  --output_report evaluation_report.json
```

---

## Model Architecture

### EfficientNet-Lite0 Dual-Output Model
```
Input: [1, 224, 224, 3]
    ↓
EfficientNet-Lite0 Backbone
    ↓
Last Conv Layer (7×7×1280 feature maps) ──→ Output 1: Feature Maps
    ↓
Global Average Pooling
    ↓
Dense(1) Classification Head ──→ Output 2: DR Logit
```

**Key Characteristics:**
- Pure ReLU6 activations (lossless INT8 quantization)
- No Squeeze-and-Excitation blocks (faster mobile inference)
- Dual outputs for GAP-CAM heatmap generation

---

## Domain Shift Mitigation

Training augmentation pipeline to simulate portable camera artifacts:
- Random rotation (0° to 360°)
- Horizontal/vertical flips
- Gaussian blur (σ = 0.5-1.5) - simulates handheld motion blur
- Brightness jitter (±25%)
- Contrast jitter (±20%)
- Random vignetting masks - simulates small-pupil artifacts

---

## Evaluation Metrics

Target screening performance:
- **Sensitivity (Recall):** ≥ 90% (minimize false negatives)
- **Specificity:** ≥ 85%
- **AUC-ROC:** ≥ 0.95

Binary classification threshold: 0.50 (tunable during validation)

---

## Next Steps (Not Started Yet)

- [ ] Download APTOS 2019 dataset
- [ ] Implement data preprocessing script
- [ ] Implement training script
- [ ] Implement TFLite export script
- [ ] Validate GAP-CAM heatmap quality
- [ ] Benchmark INT8 quantization accuracy

---

**Status:** Directory structure created, awaiting dataset acquisition and script implementation.
