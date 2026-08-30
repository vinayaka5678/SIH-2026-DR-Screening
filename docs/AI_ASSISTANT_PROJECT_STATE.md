# AI Assistant Project State
**Last Updated:** 2026-08-30  
**Project:** SIH 2026 - Offline Explainable AI for Diabetic Retinopathy Screening

---

## 1. Project Objective
Build an **offline Android application** for binary diabetic retinopathy (DR) screening (DR Present / No DR Detected) with explainable AI (Grad-CAM heatmaps) for deployment in rural and semi-urban Primary Health Centres (PHCs) in Karnataka, India.

Target users: PHC staff, trained community health workers, ASHA workers.

---

## 2. Official Requirements
- **Offline-only:** Zero network dependencies (image capture, AI inference, Grad-CAM, storage, UI)
- **Binary classification:** DR Present vs No DR Detected (NOT 5-level ICDR grading)
- **Explainable AI:** Grad-CAM heatmap overlay on fundus images (CORE requirement)
- **Target hardware:** Low-end Android devices (2-4 GB RAM, Android 8.0+)
- **Localization:** English + Kannada (ಕನ್ನಡ) UI
- **Image source:** Portable/low-cost fundus cameras (e.g., Remidio FOP, Forus 3nethra)
- **Privacy:** No patient PII storage, no cloud, no backend
- **Team:** 6 members (mostly beginners)

---

## 3. Current Architecture

### Mobile Application
- **Platform:** Native Android (Kotlin)
- **UI Framework:** Jetpack Compose (100%, no XML)
- **Architecture Pattern:** Single-Activity MVVM with Unidirectional Data Flow
- **Camera:** CameraX with `AndroidView(PreviewView)` integration
- **Database:** Room (SQLite)
- **Storage:** App-internal sandboxed filesystem

### AI/ML Pipeline
- **Training Framework:** TensorFlow / Keras (Python)
- **Model Backbone:** EfficientNet-Lite0 (primary), EfficientNet-Lite2 (high-res option)
- **Input Resolution:** 224×224 (Lite0) or 260×260 (Lite2)
- **Model Export:** Dual-output `.tflite` model (classification logit + feature maps)
- **On-Device Runtime:** Google AI Edge LiteRT (TensorFlow Lite)
- **Quantization:** INT8 Post-Training Quantization (~5.5 MB model)
- **Inference Target:** <100 ms on low-end devices

### Grad-CAM Implementation
- **Approach:** GAP-CAM (Global Average Pooling - Class Activation Mapping)
- **Mathematical Basis:** For GAP → Dense architectures, CAM weights equal Grad-CAM gradients (mathematically proven equivalence)
- **On-Device Execution:** Single forward pass + linear projection (~5 ms heatmap overhead)
- **No Backpropagation Required:** Precomputed dense layer weights exported alongside model

---

## 4. Technology Stack & Key Decisions

| Component | Technology | Justification |
|---|---|---|
| **Android Language** | Kotlin | Official Android language, coroutines for async ML |
| **UI Framework** | Jetpack Compose | Verified: CameraX integration, Canvas heatmap overlay, Kannada Unicode support on API 26+ |
| **Training Framework** | TensorFlow/Keras | Single-step `.tflite` export (no PyTorch→ONNX conversion friction), beginner-friendly |
| **Model Architecture** | EfficientNet-Lite0 | Pure ReLU6 (lossless INT8 quantization), ~5.5 MB, <85 ms inference, no SE blocks |
| **On-Device Runtime** | LiteRT (TFLite) | ~1.5 MB footprint, native multi-output support, mature quantization tooling |
| **Grad-CAM Method** | GAP-CAM Linear Projection | Exact Grad-CAM equivalence without backprop, <5 ms overhead |

**Why TensorFlow/Keras over PyTorch:**
- Direct Keras → `.tflite` export (zero conversion issues)
- Native multi-output functional model support
- 3-line INT8 quantization in `TFLiteConverter`
- Beginner-friendly high-level API (`model.fit()`, `image_dataset_from_directory()`)

**Why EfficientNet-Lite0 over MobileNetV3:**
- Pure ReLU6 activations (no Hard-Swish) → lossless INT8 quantization (<0.4% accuracy drop)
- No Squeeze-and-Excitation blocks → faster inference, better quantization
- Binary DR AUC: ~0.97 (sufficient for screening triage)

---

## 5. Current Development Phase
**Phase 0: Architecture Research & Planning** (COMPLETED)

All technical research agents completed:
- ✅ LiteRT vs ONNX Runtime Mobile comparison
- ✅ Grad-CAM on-device feasibility analysis (GAP-CAM breakthrough confirmed)
- ✅ Model architecture comparison (MobileNetV3 vs EfficientNet-Lite)
- ✅ PyTorch vs TensorFlow training framework evaluation
- ✅ Dataset domain shift analysis (tabletop vs portable cameras)
- ✅ Jetpack Compose compatibility verification

---

## 6. Completed Tasks
- [x] Environment scan (JDK 25, Python 3.13, Git, Node.js detected; Android Studio not installed yet)
- [x] Technology stack research across 6 parallel agents
- [x] Final architecture specification document prepared
- [x] Mathematical proof of GAP-CAM = Grad-CAM equivalence for GAP→Dense architectures
- [x] Domain shift mitigation strategy defined (synthetic augmentation, Indian datasets priority)
- [x] Team role assignments (6-member division)
- [x] Development phase roadmap (Phases 1-5, Weeks 1-10)
- [x] Project state documentation created

---

## 7. Pending Tasks
- [ ] User confirmation on proposed architecture
- [ ] Install Android Studio
- [ ] Initialize Android project (Kotlin + Jetpack Compose)
- [ ] Initialize Git repository
- [ ] Set up Python ML environment (TensorFlow, Keras, OpenCV)
- [ ] Download and preprocess datasets (APTOS 2019, EyePACS)
- [ ] Train EfficientNet-Lite0 dual-output model
- [ ] Convert to INT8 `.tflite` and validate quantization accuracy
- [ ] Integrate LiteRT runtime into Android app
- [ ] Implement GAP-CAM heatmap generation in Kotlin
- [ ] Implement image quality checks (blur, exposure, centering)
- [ ] Implement Kannada localization
- [ ] Build Room database schema
- [ ] Test on low-end device profiles

---

## 8. Files Created/Modified
**Created:**
- `docs/AI_ASSISTANT_PROJECT_STATE.md` (this file)
- `docs/CHANGELOG.md`

**No Android project files or ML training scripts created yet** (awaiting user confirmation).

---

## 9. Important Implementation Decisions

### Grad-CAM Breakthrough (Mathematical)
For CNN architectures where the last convolutional layer is followed by Global Average Pooling (GAP) and a single Dense/Linear layer:

```
Grad-CAM weights αₖ = (1/Z) × wₖ
```

Where `wₖ` is the Dense layer weight for channel `k`. This means:
- **No gradient computation needed on Android**
- **No dataset-averaged weights needed**
- **Exact Grad-CAM quality in a single forward pass**

Implementation:
1. Export model with 2 outputs: `[logit, feature_maps[7,7,1280]]`
2. Export Dense layer weights as `weights.json` (~5 KB)
3. On Android: `heatmap[x,y] = ReLU(Σ wₖ × feature_maps[x,y,k])`
4. Performance: ~15-30 ms inference + ~5 ms heatmap = <50 ms total

### Domain Shift Mitigation
Public datasets (EyePACS, APTOS, Messidor) use clinical tabletop cameras. Target deployment uses portable cameras (Remidio, 3nethra).

Mitigation:
1. Prioritize Indian datasets (APTOS, IDRiD)
2. Synthetic augmentation: blur (σ=0.5-1.5), brightness (±25%), vignetting, color jitter
3. Upstream image quality gate (Laplacian variance, exposure checks)
4. Binary classification (more robust than 5-level grading)
5. If possible: 50-100 local images for fine-tuning

---

## 10. Commands/Tools Used
```bash
# Environment checks performed:
java -version          # → JDK 25.0.1
python --version       # → Python 3.13.1
git --version          # → 2.54.0
node --version         # → v24.20.0
```

No Android Studio, Android SDK, or ML packages installed yet.

---

## 11. Tests Performed
None yet (no code implementation started).

---

## 12. Errors/Issues and Status
No blocking issues.

**Open Risks:**
1. **Domain shift** (tabletop → portable camera): Mitigated via augmentation + quality gate
2. **No ophthalmologist validation yet**: Acknowledged; will label as "unvalidated screening aid"
3. **No confirmed Android test devices**: Will use emulators with 2GB RAM profiles
4. **No confirmed dataset licenses**: Need to verify APTOS/EyePACS allow academic/research use

---

## 13. Dataset/Model Status
**Datasets (Not Downloaded Yet):**
- APTOS 2019 Blindness Detection (5,590 images, Indian population, Kaggle)
- EyePACS / Kaggle DR Detection 2015 (88,702 images, US population)
- IDRiD (516 images, Indian population, Maharashtra)

**Model Status:**
- Architecture selected: EfficientNet-Lite0 (224×224, INT8 ~5.5 MB)
- Backup: EfficientNet-Lite2 (260×260, INT8 ~7.0 MB) if higher resolution needed
- No training started yet

---

## 14. Important Assumptions
1. **Internet available for initial setup** (downloading Android Studio, Python packages, datasets)
2. **Team has Windows 11 PC** (confirmed: user is on Windows 11)
3. **APTOS/EyePACS datasets allow academic use** (to be verified before training)
4. **Target deployment devices run Android 8.0+** (API 26+)
5. **PHC staff can operate a smartphone camera app** (basic digital literacy assumed)
6. **No MATLAB, Simulink, or 5-level ICDR classification required** (explicitly excluded)

---

## 15. Decisions Requiring User Approval
🔴 **AWAITING USER CONFIRMATION:**

1. **Approval of the complete architecture** (Sections A-Q in the specification)
2. **Confirmation on team roles** (6-member division in Section N)
3. **Approval of development phases** (Phases 1-5, 10-week timeline in Section M)
4. **Permission to proceed to Phase 1**: Install Android Studio, initialize project, set up Git

**No implementation work will start until user confirms.**

---

## 16. Exact Next Recommended Step

**After User Confirmation:**

**Step 1:** Install Android Studio
- Download Android Studio Ladybug (2024.2.1) or later
- Install Android SDK API 26 (Android 8.0) minimum
- Install Android SDK API 35 (Android 15) target
- Install Android Emulator and configure low-RAM profile (2 GB RAM, API 26)

**Step 2:** Initialize Android Project
```bash
# Via Android Studio New Project wizard:
# - Template: Empty Activity (Compose)
# - Language: Kotlin
# - Minimum SDK: API 26 (Android 8.0)
# - Build configuration: Kotlin DSL (build.gradle.kts)
```

**Step 3:** Initialize Git Repository
```bash
cd C:\Users\vinay\SIH-2026-DR-Screening
git init
git add .
git commit -m "Initial commit: Project structure

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

**Step 4:** Set Up Python ML Environment
```bash
python -m venv venv_ml
.\venv_ml\Scripts\activate
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn
```

---

**STATUS:** Waiting for user confirmation to proceed to Phase 1.
