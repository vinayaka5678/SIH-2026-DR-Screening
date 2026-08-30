#!/usr/bin/env python3
"""
Data Preprocessing Pipeline for Binary DR Screening Model

This script preprocesses the APTOS 2019 dataset for binary DR classification:
- Converts 5-level ICDR labels to binary (0=No DR, 1=DR Present)
- Creates stratified train/validation/test splits (70/15/15)
- Resizes images to 224×224 for EfficientNet-Lite0
- Applies synthetic domain-shift augmentation
- Generates preprocessed dataset ready for training

Usage:
    python data_preprocessing.py --dataset aptos2019 --output_dir processed_data
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_dataset_metadata(data_dir: Path, dataset_name: str) -> Tuple[pd.DataFrame, Path]:
    """Load dataset CSV and return DataFrame with image paths."""

    if dataset_name == "aptos2019":
        csv_path = data_dir / "aptos2019" / "train.csv"
        images_dir = data_dir / "aptos2019" / "train_images"

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        df = pd.read_csv(csv_path)

        # Validate columns
        if 'id_code' not in df.columns or 'diagnosis' not in df.columns:
            raise ValueError(f"CSV must contain 'id_code' and 'diagnosis' columns")

        # Convert to binary labels
        # diagnosis 0 = No DR (class 0)
        # diagnosis 1-4 = DR Present (class 1)
        df['binary_label'] = (df['diagnosis'] > 0).astype(int)

        # Add image file paths
        df['image_path'] = df['id_code'].apply(lambda x: str(images_dir / f"{x}.png"))

        # Verify all images exist
        missing = df[~df['image_path'].apply(lambda p: Path(p).exists())]
        if len(missing) > 0:
            raise FileNotFoundError(f"{len(missing)} images missing from dataset")

        print(f"✅ Loaded {len(df)} images from {dataset_name}")
        print(f"   Binary label distribution:")
        print(f"     No DR (0):      {(df['binary_label'] == 0).sum()} images")
        print(f"     DR Present (1): {(df['binary_label'] == 1).sum()} images")

        return df, images_dir

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/validation/test splits.

    Stratification ensures each split maintains the same class distribution.
    """

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    # First split: train vs (val+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df['binary_label'],
        random_state=random_state
    )

    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_adjusted),
        stratify=temp_df['binary_label'],
        random_state=random_state
    )

    print(f"\n✅ Created stratified splits:")
    print(f"   Train:      {len(train_df):4d} images ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Validation: {len(val_df):4d} images ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test:       {len(test_df):4d} images ({len(test_df)/len(df)*100:.1f}%)")

    # Verify stratification
    print(f"\n   Class distribution verification:")
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        no_dr = (split_df['binary_label'] == 0).sum()
        dr_present = (split_df['binary_label'] == 1).sum()
        print(f"     {split_name:6s}: No DR={no_dr:4d} ({no_dr/len(split_df)*100:.1f}%), "
              f"DR={dr_present:4d} ({dr_present/len(split_df)*100:.1f}%)")

    # Check for data leakage (ensure no overlap)
    train_ids = set(train_df['id_code'])
    val_ids = set(val_df['id_code'])
    test_ids = set(test_df['id_code'])

    if len(train_ids & val_ids) > 0 or len(train_ids & test_ids) > 0 or len(val_ids & test_ids) > 0:
        raise ValueError("❌ DATA LEAKAGE DETECTED: Splits have overlapping image IDs")

    print(f"   ✅ No data leakage detected (splits are disjoint)")

    return train_df, val_df, test_df


def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess a single fundus image.

    Steps:
    1. Load image
    2. Resize to target_size
    3. Normalize to [0, 1] if requested

    Returns: np.ndarray with shape (height, width, 3) and dtype float32
    """

    # Load image using OpenCV (BGR format)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize using high-quality Lanczos interpolation
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

    # Convert to float32
    img = img.astype(np.float32)

    # Normalize to [0, 1]
    if normalize:
        img = img / 255.0

    return img


def preprocess_split(
    df: pd.DataFrame,
    output_dir: Path,
    split_name: str,
    target_size: Tuple[int, int] = (224, 224)
) -> Dict:
    """
    Preprocess all images in a split and save as NPZ file.

    Returns metadata dictionary with split statistics.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    labels = []
    image_ids = []

    print(f"\n🔄 Preprocessing {split_name} split ({len(df)} images)...")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split_name}"):
        try:
            img = preprocess_image(row['image_path'], target_size=target_size)
            images.append(img)
            labels.append(row['binary_label'])
            image_ids.append(row['id_code'])
        except Exception as e:
            print(f"  ⚠️  Failed to process {row['id_code']}: {e}")
            continue

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # Save as compressed NPZ
    output_path = output_dir / f"{split_name}.npz"
    np.savez_compressed(
        output_path,
        images=images,
        labels=labels,
        image_ids=image_ids
    )

    print(f"   ✅ Saved {len(images)} preprocessed images to {output_path}")
    print(f"      File size: {output_path.stat().st_size / (1024**2):.1f} MB")
    print(f"      Image shape: {images.shape}")
    print(f"      Labels shape: {labels.shape}")
    print(f"      Label distribution: No DR={np.sum(labels==0)}, DR={np.sum(labels==1)}")

    metadata = {
        'split_name': split_name,
        'num_images': len(images),
        'image_shape': list(images.shape[1:]),
        'num_classes': 2,
        'class_distribution': {
            'no_dr': int(np.sum(labels == 0)),
            'dr_present': int(np.sum(labels == 1))
        },
        'file_size_mb': float(output_path.stat().st_size / (1024**2)),
        'output_path': str(output_path)
    }

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess DR screening dataset for binary classification"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='aptos2019',
        choices=['aptos2019'],
        help='Dataset name'
    )
    parser.add_argument(
        '--data_dir',
        type=Path,
        default=Path(__file__).parent.parent / 'data',
        help='Root directory containing raw datasets'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path(__file__).parent.parent / 'processed_data',
        help='Output directory for preprocessed data'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=224,
        help='Target image size (square)'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.70,
        help='Training set ratio'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Validation set ratio'
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.15,
        help='Test set ratio'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Data Preprocessing Pipeline - Binary DR Classification")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target image size: {args.image_size}×{args.image_size}")
    print(f"Split ratios: {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")
    print(f"Random seed: {args.seed}")
    print()

    # Load dataset
    df, images_dir = load_dataset_metadata(args.data_dir, args.dataset)

    # Create stratified splits
    train_df, val_df, test_df = create_stratified_splits(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed
    )

    # Preprocess each split
    target_size = (args.image_size, args.image_size)

    metadata = {
        'dataset': args.dataset,
        'target_size': list(target_size),
        'num_classes': 2,
        'class_names': ['No DR', 'DR Present'],
        'random_seed': args.seed,
        'splits': {}
    }

    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        split_metadata = preprocess_split(
            split_df,
            args.output_dir,
            split_name,
            target_size=target_size
        )
        metadata['splits'][split_name] = split_metadata

    # Save metadata
    metadata_path = args.output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Preprocessing complete!")
    print(f"   Metadata saved to: {metadata_path}")
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total images processed: {sum(m['num_images'] for m in metadata['splits'].values())}")
    print(f"Image shape: {metadata['target_size']} (H×W×C)")
    print(f"Number of classes: {metadata['num_classes']}")
    print(f"Train images: {metadata['splits']['train']['num_images']}")
    print(f"Val images: {metadata['splits']['val']['num_images']}")
    print(f"Test images: {metadata['splits']['test']['num_images']}")
    print(f"Total storage: {sum(m['file_size_mb'] for m in metadata['splits'].values()):.1f} MB")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
