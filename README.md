# Offline Explainable AI for Diabetic Retinopathy Screening

**SIH 2026 Project**  
**Target Deployment:** Rural and Semi-Urban Primary Health Centres (PHCs) in Karnataka, India

---

## Project Overview

An **offline Android application** for binary diabetic retinopathy (DR) screening with explainable AI visualization (Grad-CAM heatmaps). Designed for deployment on low-end Android devices (2-4 GB RAM) in areas with limited or no internet connectivity.

### Key Features
- ✅ **Binary DR Classification:** DR Present / No DR Detected
- ✅ **Explainable AI:** Grad-CAM heatmap overlays showing model attention regions
- ✅ **100% Offline:** Zero network dependencies (inference, storage, UI)
- ✅ **Bilingual UI:** English + Kannada (ಕನ್ನಡ)
- ✅ **Image Quality Assessment:** Pre-inference checks for blur, exposure, and framing
- ✅ **Privacy-First:** No patient PII storage, no cloud backend
- ✅ **Low-End Device Optimized:** <100 ms inference on budget Android phones

---

## Technology Stack

### Mobile Application
- **Platform:** Native Android (Kotlin)
- **UI Framework:** Jetpack Compose
- **Architecture:** Single-Activity MVVM
- **Camera:** CameraX
- **Database:** Room (SQLite)
- **Min SDK:** API 26 (Android 8.0)

### AI/ML Pipeline
- **Training Framework:** TensorFlow / Keras (Python)
- **Model Architecture:** EfficientNet-Lite0 (224×224, INT8 quantized ~5.5 MB)
- **On-Device Runtime:** LiteRT (TensorFlow Lite)
- **Explainability:** GAP-CAM (mathematically equivalent to Grad-CAM)

---

## Project Structure

```
SIH-2026-DR-Screening/
├── docs/                          # Project documentation
│   ├── AI_ASSISTANT_PROJECT_STATE.md
│   ├── CHANGELOG.md
│   └── architecture/              # Architecture diagrams and specs
├── android/                       # Android application (to be created)
├── ml_training/                   # Python ML training pipeline
│   ├── notebooks/                 # Jupyter notebooks for experimentation
│   ├── src/                       # Training scripts
│   ├── data/                      # Dataset storage (not tracked)
│   ├── requirements.txt
│   └── README.md
├── .gitignore
└── README.md
```

---

## Development Phases

### Phase 0: Architecture & Planning ✅ (Completed)
- Technology stack research
- Architecture specification
- Team role assignments

### Phase 1: Environment Setup 🔄 (In Progress)
- Android Studio verification
- Git repository initialization
- Project structure creation

### Phase 2: AI Training & Conversion (Weeks 3-5)
- Dataset acquisition (APTOS 2019, EyePACS)
- EfficientNet-Lite0 training with transfer learning
- Dual-output model export & INT8 quantization

### Phase 3: Android AI Integration (Weeks 5-7)
- LiteRT runtime integration
- GAP-CAM heatmap generation
- Image quality assessment

### Phase 4: Localization & Polish (Weeks 7-8)
- Kannada localization
- Room database implementation
- UI refinement

### Phase 5: Testing & Demo Prep (Weeks 9-10)
- Device testing & benchmarking
- Documentation
- SIH presentation preparation

---

## Team Structure (6 Members)

| Member | Role |
|---|---|
| **Member 1** | Android Architecture Lead |
| **Member 2** | Android UI/UX & Localization |
| **Member 3** | AI/ML Training Lead |
| **Member 4** | Data Engineering & Model Conversion |
| **Member 5** | Edge AI Integration & Visualization |
| **Member 6** | QA, Testing & Documentation |

---

## Getting Started

### Prerequisites
- **Android Studio:** Ladybug (2024.2.1) or later ✅
- **JDK:** 17 or 21 ✅ (JDK 25.0.1 installed)
- **Python:** 3.11 - 3.13 ✅ (Python 3.13.1 installed)
- **Git:** Latest version ✅ (Git 2.54.0 installed)
- **Android SDK:** API 26 (minimum), API 35 (target) ✅

### Environment Setup

#### 1. Clone or Navigate to Repository
```bash
cd C:\Users\vinay\SIH-2026-DR-Screening
```

#### 2. Set Up Python ML Environment
```bash
cd ml_training
python -m venv venv_ml
.\venv_ml\Scripts\activate  # Windows
pip install -r requirements.txt
```

#### 3. Open Android Project (When Created)
- Open Android Studio
- File → New → Project → Empty Activity (Compose)
- Create in `android/` directory

---

## Datasets

### Training Datasets (Not Included)
- **APTOS 2019 Blindness Detection** (Kaggle)
- **EyePACS / Kaggle DR Detection 2015**
- **IDRiD** (Indian Diabetic Retinopathy Image Dataset)

**Note:** Datasets must be downloaded separately due to size and licensing.

---

## Key Technical Decisions

### Why TensorFlow/Keras over PyTorch?
- Single-step `.tflite` export (no multi-hop ONNX conversion)
- Native multi-output model support
- Beginner-friendly high-level API
- 3-line INT8 quantization

### Why EfficientNet-Lite0?
- Pure ReLU6 activations → lossless INT8 quantization
- No Squeeze-and-Excitation blocks → faster inference
- ~5.5 MB quantized model size
- <85 ms inference on low-end devices

### GAP-CAM: Grad-CAM Without Backpropagation
For architectures with Global Average Pooling → Dense layers, Class Activation Mapping (CAM) weights equal Grad-CAM gradients:

```
Heatmap(x,y) = ReLU(Σ wₖ × FeatureMap[x,y,k])
```

Where `wₖ` are the final Dense layer weights. This provides exact Grad-CAM quality in a single forward pass with <5 ms overhead.

---

## Important Constraints

### What This Project IS:
✅ Binary DR screening (DR Present / No DR Detected)  
✅ Explainable AI with Grad-CAM heatmaps  
✅ Offline-first mobile application  
✅ English + Kannada localization  
✅ Image quality assessment  

### What This Project IS NOT:
❌ 5-level ICDR severity grading  
❌ Lesion segmentation or detection  
❌ Cloud-based or web application  
❌ Diagnostic medical device (screening aid only)  
❌ Clinically validated system (requires ophthalmologist validation)  

---

## Disclaimer

This application is an **automated screening aid for primary healthcare triage** and does NOT provide a definitive medical diagnosis. All screening results must be confirmed by a qualified ophthalmologist. The model has not yet undergone clinical validation.

---

**Last Updated:** 2026-08-30
