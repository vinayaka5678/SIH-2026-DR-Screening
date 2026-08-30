# Changelog
All notable changes to the SIH 2026 DR Screening project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Phase 0: Architecture Research & Planning

#### [2026-08-30] - Architecture Finalized

**Added**
- Complete architectural specification (Sections A-Q)
- Technology stack decisions documented
- 6-member team role assignments
- 10-week development phase roadmap (Phases 1-5)
- Project state documentation (`AI_ASSISTANT_PROJECT_STATE.md`)
- Project changelog (`CHANGELOG.md`)

**Research Completed**
- LiteRT (TensorFlow Lite) vs ONNX Runtime Mobile comparison → **Selected: LiteRT**
- TensorFlow/Keras vs PyTorch training framework evaluation → **Selected: TensorFlow/Keras**
- Model architecture comparison (MobileNetV3 vs EfficientNet-Lite family) → **Selected: EfficientNet-Lite0**
- Grad-CAM on-device implementation feasibility → **Selected: GAP-CAM (mathematically equivalent to Grad-CAM)**
- Jetpack Compose compatibility verification → **Confirmed: 100% Compose viable**
- Dataset domain shift analysis (tabletop cameras vs portable cameras in rural Karnataka)

**Architectural Decisions**
- **Mobile Platform:** Native Android (Kotlin) with Jetpack Compose
- **AI Training:** TensorFlow/Keras (Python) for single-step `.tflite` export
- **Model:** EfficientNet-Lite0 (224×224, INT8 ~5.5 MB, <85 ms inference)
- **On-Device Runtime:** Google AI Edge LiteRT (TensorFlow Lite)
- **Grad-CAM Method:** GAP-CAM linear projection (exact Grad-CAM equivalence, <5 ms overhead)
- **Localization:** English + Kannada via Android resource system
- **Privacy:** Zero network permissions, local-only storage, no PII

**Technical Breakthrough**
- Mathematical proof: For Global Average Pooling (GAP) → Dense architectures, Class Activation Mapping (CAM) using final Dense layer weights produces **identical results to Grad-CAM** without requiring backpropagation or gradient computation on-device.

**Datasets Identified (Not Downloaded Yet)**
- APTOS 2019 Blindness Detection (5,590 images, Indian population)
- EyePACS / Kaggle DR Detection 2015 (88,702 images, US population)
- IDRiD (516 images, Indian population, pixel-level annotations)

**Environment Verified**
- JDK 25.0.1 installed ✓
- Python 3.13.1 installed ✓
- Git 2.54.0 installed ✓
- Node.js v24.20.0 installed ✓
- Android Studio: Not installed yet
- Android SDK: Not installed yet

**Status**
- Phase 0 (Architecture & Planning): **COMPLETED**
- Phase 1 (Environment & Project Setup): **IN PROGRESS**

#### [2026-08-30] - Phase 1 Started

**Environment Verified**
- Android Studio: Installed at `C:\Program Files\Android\Android Studio`
- Android SDK: Installed at `C:\Users\vinay\AppData\Local\Android\Sdk`
  - Platforms: API 33, 34, 36
  - Build tools: 35.0.0, 36.1.0, 37.0.0
  - Platform tools: ADB available
- JDK 25.0.1 installed
- Python 3.13.1 installed
- Git 2.54.0 installed
- Node.js v24.20.0 installed

**Added**
- Git repository initialized (commit `ed8fef8`)
- `.gitignore` (Android, Python, ML, IDE exclusions)
- `README.md` (project overview, getting started guide)
- `ml_training/` directory structure (notebooks/, src/, data/)
- `ml_training/requirements.txt` (TensorFlow, Keras, OpenCV, etc.)
- `ml_training/README.md` (training pipeline documentation)
- `docs/architecture/` directory

**Pending (Phase 1)**
- Android project creation in Android Studio
- Python virtual environment setup (venv_ml)
- Android emulator configuration

**Status**
- User approved architecture and Phase 1
- Git repository initialized with initial commit
- Project structure created
- Android project created successfully

#### [2026-08-31] - Android Project Created

**Added**
- Android project structure (native Kotlin + Jetpack Compose)
- `android/app/build.gradle.kts` - Build configuration with all dependencies
- `android/gradle/libs.versions.toml` - Version catalog (Compose BOM, TFLite, CameraX, Room)
- `MainActivity.kt` - Single Activity with temporary Compose home screen
- `ui/theme/` - Material 3 theme (Color.kt, Theme.kt, Type.kt)
- `AndroidManifest.xml` - Camera permission, NO internet permission
- `values/strings.xml` - English localization
- `values-kn/strings.xml` - Kannada (ಕನ್ನಡ) localization
- `proguard-rules.pro` - ProGuard rules for TFLite and Room
- `android/README.md` - Android project documentation

**Dependencies Added**
- Jetpack Compose BOM 2024.12.01
- TensorFlow Lite 2.14.0 + Support
- CameraX 1.4.1 (Core, Camera2, Lifecycle, View)
- Room 2.6.1 (Runtime, KTX)
- Navigation Compose 2.8.5
- Coroutines 1.9.0

**Git Commits**
- `59e689e` - Add Android project structure
- `[latest]` - Update MainActivity and localization

**Status**
- Phase 1: **COMPLETED**
- Phase 2: **IN PROGRESS** (Environment setup completed)

---

## [Unreleased]

### Phase 2: AI Training & Model Conversion (In Progress)

#### [2026-08-31] - Python ML Environment Setup Completed

**Environment Created**
- Python virtual environment: `ml_training/venv_ml/`
- Python 3.13.1
- Total packages installed: 150+
- Virtual environment size: ~1.2 GB

**Core ML Packages Installed**
- TensorFlow 2.21.0
- Keras 3.15.1
- NumPy 2.5.2
- Pandas 3.0.5
- SciPy 1.18.1
- Scikit-learn 1.9.0

**Image Processing**
- OpenCV 5.0.0.93
- Pillow 12.3.0
- Albumentations 2.0.8 (augmentation)

**Visualization & Development**
- Matplotlib 3.11.1
- Seaborn 0.13.2
- Jupyter Lab 4.6.3
- Jupyter Notebook 7.6.2
- tf-keras-vis 0.8.7 (Grad-CAM visualization)

**Documentation Created**
- `ml_training/DATASET_DOWNLOAD.md` - Manual dataset download instructions for APTOS 2019, IDRiD, EyePACS
- `ml_training/PHASE2_PROGRESS.md` - Phase 2 progress tracking
- `ml_training/src/verify_dataset.py` - Dataset integrity verification script

**TensorFlow Configuration**
- TensorFlow 2.21.0 with oneDNN optimization
- GPU: Not available on native Windows (WSL2 or TensorFlow-DirectML required)
- Training: Will use Google Colab with GPU for model training
- Local: CPU-only for development and testing

**Status**
- Phase 2A (Environment Setup): ✅ COMPLETED
- Phase 2B (Dataset Verification): ✅ COMPLETED (APTOS 2019, 3,662 images verified)
- Phase 2C (Training Scripts & Pipeline): ✅ COMPLETED (Lightweight tests passed)
- Phase 2D (Model Training): ⏳ PENDING (Awaiting user approval)
- Phase 2E (Model Conversion): ⏳ PENDING

#### [2026-08-31] - Phase 2C: Training Pipeline Scripts & Google Colab Notebook Completed

**Added**
- `ml_training/src/data_preprocessing.py` - Binary label conversion, stratified 70/15/15 split, Lanczos4 224×224 resize, NPZ export
- `ml_training/src/train_model.py` - EfficientNet-Lite0 dual-output architecture (classification + GAP-CAM feature maps), synthetic domain shift augmentation, class weighting, callbacks
- `ml_training/src/export_tflite.py` - INT8 post-training quantization, dense layer weight extraction for zero-backprop GAP-CAM, deployment bundle metadata
- `ml_training/notebooks/train_dr_model.ipynb` - Google Colab GPU-accelerated training workflow
- `ml_training/src/test_pipeline_lightweight.py` - End-to-end pipeline validation on 50 sample images
- `ml_training/PHASE2C_REPORT.md` - Phase 2C comprehensive architecture & test report

**Validation Performed**
- Binary label conversion verified across all 5 ICDR grades (0→0, 1-4→1)
- Stratified splitting verified: 2,563 train (70.0%), 549 val (15.0%), 550 test (15.0%)
- Disjoint set verification: 0% overlap, zero data leakage
- Dual-output model compilation & single-epoch pass verified on CPU
- GAP-CAM dense weights extraction verified (1280 weights extracted, heatmap calculated)
- INT8 quantization verified: 6.82 MB TFLite model generated, uint8 input/output validated

---

## Template for Future Entries

### Phase N: [Phase Name]

#### [YYYY-MM-DD] - [Brief Description]

**Added**
- New features, files, or capabilities

**Changed**
- Modifications to existing functionality

**Fixed**
- Bug fixes

**Removed**
- Deprecated or removed features

**Technical Details**
- Important implementation notes

**Performance**
- Benchmarks or performance measurements

**Testing**
- Tests performed and results

**Status**
- Current state of this phase
