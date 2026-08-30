# Dataset Download Instructions
**SIH 2026 - DR Screening Model Training**

---

## Required Datasets

### 1. APTOS 2019 Blindness Detection (PRIMARY)

**Source:** Kaggle Competition  
**URL:** https://www.kaggle.com/c/aptos2019-blindness-detection/data

**Files to Download:**
- `train.csv` (~119 KB) - Image IDs and DR severity labels (0-4)
- `train_images.zip` (~3.5 GB) - 3,662 training fundus images
- `test_images.zip` (~1.8 GB) - 1,928 test images (optional for validation)

**Manual Download Steps:**
1. Visit: https://www.kaggle.com/c/aptos2019-blindness-detection/data
2. Click "Download All" or download individual files
3. Extract to: `ml_training/data/aptos2019/`
   ```
   ml_training/data/aptos2019/
   ├── train.csv
   └── train_images/
       ├── 000c1434d8d7.png
       ├── 001639a390f0.png
       └── ... (3,662 images)
   ```

**Dataset License:**
- Competition rules: https://www.kaggle.com/c/aptos2019-blindness-detection/rules
- Verify that academic/research use is permitted

**Population:** Indian (Aravind Eye Hospital, Tamil Nadu)  
**Camera Type:** Fundus camera (portable + clinical mix)  
**Grading:** 5-level ICDR scale (0 = No DR, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = PDR)

**For Binary Classification:**
- Class 0 (No DR): Labels 0
- Class 1 (DR Present): Labels 1, 2, 3, 4

---

### 2. EyePACS / Kaggle DR Detection 2015 (OPTIONAL - LARGE)

**Source:** Kaggle Competition  
**URL:** https://www.kaggle.com/c/diabetic-retinopathy-detection/data

**Files to Download:**
- `trainLabels.csv` (~2 MB) - 35,126 training labels
- `train.zip` (~35 GB compressed, ~88 GB uncompressed) - Training images
- `test.zip` (~53 GB compressed) - Test images

**Manual Download Steps:**
1. Visit: https://www.kaggle.com/c/diabetic-retinopathy-detection/data
2. Download `trainLabels.csv` and `train.zip`
3. Extract to: `ml_training/data/eyepacs/`

**⚠️ WARNING:** This dataset is very large (88 GB). Only download if:
- You have sufficient storage space
- You need additional training data volume
- APTOS 2019 alone is insufficient

**Population:** Mixed US (safety-net clinics)  
**Camera Type:** Clinical tabletop fundus cameras  

**Recommendation:** Start with APTOS 2019 only. Add EyePACS later if needed for improved accuracy.

---

### 3. IDRiD (Indian Diabetic Retinopathy Image Dataset) (OPTIONAL - SMALL)

**Source:** IEEE DataPort  
**URL:** https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid

**Files to Download:**
- Training set: 413 images
- Testing set: 103 images
- Segmentation masks (lesion annotations)

**Manual Download Steps:**
1. Visit: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
2. Create account (free) and download dataset
3. Extract to: `ml_training/data/idrid/`

**Population:** Indian (Eye Clinic, Nanded, Maharashtra)  
**Camera Type:** Kowa VX-10α (clinical tabletop)  
**Resolution:** Very high (4288×2848 px)  
**Special Feature:** Includes pixel-level lesion annotations

**Use Case:** Validation on Indian population with high-quality annotations

---

## Recommended Download Strategy

### Phase 2A: Minimum Viable Training
**Download:** APTOS 2019 only (~3.5 GB)  
**Reason:** Indian population, sufficient size for binary classification, manageable download

### Phase 2B: Enhanced Training (If Accuracy Insufficient)
**Add:** IDRiD (~500 MB)  
**Reason:** Indian population, high-quality lesion annotations for validation

### Phase 2C: Maximum Training (If Still Insufficient)
**Add:** EyePACS subset (~10 GB sample)  
**Reason:** Massive volume for transfer learning, but US population (domain shift risk)

---

## Storage Requirements

| Dataset | Compressed | Uncompressed | Recommended |
|---|---|---|---|
| **APTOS 2019** | ~3.5 GB | ~4 GB | ✅ Required |
| **IDRiD** | ~500 MB | ~600 MB | ⭐ Recommended |
| **EyePACS** | ~35 GB | ~88 GB | ⚠️ Optional |

**Total (APTOS + IDRiD):** ~4.6 GB

---

## Alternative: Use Pre-Downloaded Subsets

If storage or download bandwidth is limited, you can:
1. Download a **stratified sample** from EyePACS (e.g., 5,000 images) instead of the full 88K
2. Use only APTOS 2019 for training
3. Validate on IDRiD

**Sample Script (to be created):**
- `ml_training/scripts/download_datasets.sh` - Automated download helper
- `ml_training/scripts/sample_eyepacs.py` - Create stratified subset

---

## Dataset Directory Structure (Target)

```
ml_training/data/
├── aptos2019/
│   ├── train.csv
│   └── train_images/
│       ├── 000c1434d8d7.png
│       └── ... (3,662 images)
├── idrid/
│   ├── train/
│   │   ├── images/
│   │   └── labels.csv
│   └── test/
│       ├── images/
│       └── labels.csv
└── eyepacs/  (optional)
    ├── trainLabels.csv
    └── train/
        ├── 10_left.jpeg
        └── ... (35,126 images)
```

---

## License Verification Checklist

Before downloading:
- [ ] APTOS 2019: Read competition rules, confirm academic use permitted
- [ ] IDRiD: IEEE DataPort license allows research use
- [ ] EyePACS: Kaggle competition rules allow academic use

**⚠️ IMPORTANT:** These datasets contain medical images. Ensure compliance with:
- Dataset licenses
- Academic research guidelines
- No redistribution without permission
- No commercial use without explicit license

---

## Next Steps

1. **Download APTOS 2019** manually from Kaggle (~3.5 GB)
2. Extract to `ml_training/data/aptos2019/`
3. Run data verification script (to be created): `python src/verify_dataset.py`
4. Proceed with preprocessing and training

**Status:** Awaiting manual dataset download before proceeding with Phase 2 training pipeline.
