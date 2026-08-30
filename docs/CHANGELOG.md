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
- Awaiting user confirmation to proceed to Phase 1 (Environment & Project Setup)

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
