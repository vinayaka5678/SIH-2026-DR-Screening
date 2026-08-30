#!/usr/bin/env python3
"""
Verify downloaded datasets for DR Screening model training.

This script checks:
- Dataset directory structure
- CSV file integrity
- Image file counts
- Image readability
- Label distribution

Usage:
    python verify_dataset.py --dataset aptos2019
    python verify_dataset.py --dataset idrid
    python verify_dataset.py --dataset eyepacs
"""

import os
import sys
import pandas as pd
from pathlib import Path
from PIL import Image
import argparse


def verify_aptos2019(data_dir: Path):
    """Verify APTOS 2019 dataset structure and integrity."""
    print("=" * 60)
    print("APTOS 2019 Blindness Detection Dataset Verification")
    print("=" * 60)

    aptos_dir = data_dir / "aptos2019"

    # Check directory exists
    if not aptos_dir.exists():
        print(f"❌ Directory not found: {aptos_dir}")
        print(f"   Please download APTOS 2019 dataset from Kaggle")
        return False

    # Check train.csv
    csv_path = aptos_dir / "train.csv"
    if not csv_path.exists():
        print(f"❌ train.csv not found")
        return False

    print(f"✅ Found: {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"✅ CSV loaded: {len(df)} records")
    print(f"   Columns: {list(df.columns)}")

    # Check label distribution
    if 'diagnosis' in df.columns:
        print(f"\n📊 Label Distribution:")
        label_counts = df['diagnosis'].value_counts().sort_index()
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   Grade {label}: {count:4d} images ({percentage:5.2f}%)")

        # Binary classification distribution
        df['binary'] = (df['diagnosis'] > 0).astype(int)
        binary_counts = df['binary'].value_counts()
        print(f"\n📊 Binary Classification:")
        print(f"   No DR (0):      {binary_counts.get(0, 0):4d} images")
        print(f"   DR Present (1): {binary_counts.get(1, 0):4d} images")

    # Check train_images directory
    images_dir = aptos_dir / "train_images"
    if not images_dir.exists():
        print(f"❌ train_images/ directory not found")
        return False

    print(f"\n✅ Found: {images_dir}")

    # Count images
    image_files = list(images_dir.glob("*.png"))
    print(f"✅ Found {len(image_files)} PNG images")

    if len(image_files) != len(df):
        print(f"⚠️  WARNING: CSV has {len(df)} records but found {len(image_files)} images")

    # Test read first few images
    print(f"\n🔍 Testing image readability (first 5):")
    for i, img_path in enumerate(image_files[:5]):
        try:
            img = Image.open(img_path)
            print(f"   ✅ {img_path.name}: {img.size} {img.mode}")
        except Exception as e:
            print(f"   ❌ {img_path.name}: {e}")

    print(f"\n✅ APTOS 2019 dataset verification PASSED")
    return True


def verify_idrid(data_dir: Path):
    """Verify IDRiD dataset structure."""
    print("=" * 60)
    print("IDRiD Dataset Verification")
    print("=" * 60)

    idrid_dir = data_dir / "idrid"

    if not idrid_dir.exists():
        print(f"⚠️  IDRiD directory not found: {idrid_dir}")
        print(f"   This dataset is optional. Skipping.")
        return True

    print(f"✅ Found: {idrid_dir}")

    # Check for train/test splits
    train_dir = idrid_dir / "train" / "images"
    test_dir = idrid_dir / "test" / "images"

    if train_dir.exists():
        train_images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
        print(f"✅ Training images: {len(train_images)}")
    else:
        print(f"⚠️  No train/images/ directory")

    if test_dir.exists():
        test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
        print(f"✅ Test images: {len(test_images)}")
    else:
        print(f"⚠️  No test/images/ directory")

    print(f"\n✅ IDRiD dataset verification PASSED")
    return True


def verify_eyepacs(data_dir: Path):
    """Verify EyePACS dataset structure."""
    print("=" * 60)
    print("EyePACS/Kaggle DR Detection 2015 Verification")
    print("=" * 60)

    eyepacs_dir = data_dir / "eyepacs"

    if not eyepacs_dir.exists():
        print(f"⚠️  EyePACS directory not found: {eyepacs_dir}")
        print(f"   This dataset is optional (88 GB). Skipping.")
        return True

    print(f"✅ Found: {eyepacs_dir}")

    # Check trainLabels.csv
    csv_path = eyepacs_dir / "trainLabels.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"✅ trainLabels.csv: {len(df)} records")
    else:
        print(f"⚠️  trainLabels.csv not found")

    # Check train directory
    train_dir = eyepacs_dir / "train"
    if train_dir.exists():
        image_files = list(train_dir.glob("*.jpeg")) + list(train_dir.glob("*.jpg"))
        print(f"✅ Training images: {len(image_files)}")
    else:
        print(f"⚠️  No train/ directory")

    print(f"\n✅ EyePACS dataset verification PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify DR screening datasets")
    parser.add_argument(
        "--dataset",
        choices=["aptos2019", "idrid", "eyepacs", "all"],
        default="all",
        help="Which dataset to verify"
    )
    args = parser.parse_args()

    # Get data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"

    print(f"\n📂 Data directory: {data_dir.absolute()}\n")

    if not data_dir.exists():
        print(f"❌ Data directory does not exist: {data_dir}")
        print(f"   Creating directory...")
        data_dir.mkdir(parents=True, exist_ok=True)

    success = True

    if args.dataset == "all" or args.dataset == "aptos2019":
        success = verify_aptos2019(data_dir) and success
        print()

    if args.dataset == "all" or args.dataset == "idrid":
        success = verify_idrid(data_dir) and success
        print()

    if args.dataset == "all" or args.dataset == "eyepacs":
        success = verify_eyepacs(data_dir) and success
        print()

    if success:
        print("=" * 60)
        print("✅ All dataset verifications PASSED")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        print("❌ Some dataset verifications FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
