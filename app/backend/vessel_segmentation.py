"""
Retinal Blood Vessel Segmentation using Morphological Top-Hat Enhancement.

This module implements clinical-grade vessel segmentation using morphological image processing.
The approach is based on proven techniques from medical imaging literature for extracting
dark tubular structures (blood vessels) from retinal fundus photographs.

Key features:
- Green channel extraction and preprocessing
- Morphological illumination normalization
- CLAHE contrast enhancement
- Top-hat filtering for dark tubular structure detection
- Connected-component noise filtering
- Retinal FOV masking for clean boundaries
- Binary white/black output

References:
- Morphological vessel enhancement is standard in retinal image analysis
- Used in clinical DR screening systems worldwide
"""

import os
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage


def preprocess_for_segmentation(image_array: np.ndarray) -> np.ndarray:
    """
    Preprocess image for vessel segmentation.

    Algorithm:
    1. Extract green channel (best contrast for retinal vessels)
    2. Normalize illumination using morphological background subtraction
    3. Apply CLAHE for local contrast enhancement
    4. Return enhanced grayscale image

    Args:
        image_array: RGB image as uint8 numpy array (H x W x 3)

    Returns:
        Preprocessed grayscale image as uint8 (H x W)
    """
    # Extract green channel (vessels have best contrast here)
    green = image_array[:, :, 1].astype(np.float32)

    # Normalize illumination using morphological background subtraction
    # This corrects for uneven illumination in fundus images
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    background = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel_bg)
    illumination_normalized = cv2.subtract(green, background * 0.7)
    illumination_normalized = np.clip(illumination_normalized, 0, 255).astype(np.uint8)

    # Apply CLAHE (Contrast-Limited Adaptive Histogram Equalization)
    # for local contrast enhancement while avoiding over-amplification of noise
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(illumination_normalized)

    return enhanced


def infer_vessel_segmentation(image_path: str) -> np.ndarray:
    """
    Run morphological vessel segmentation on retinal fundus image.

    This function:
    1. Loads the image and resizes to standard resolution
    2. Preprocesses using green channel + CLAHE + illumination normalization
    3. Applies morphological top-hat for dark vessel extraction
    4. Thresholds to create vessel mask
    5. Filters noise via connected-component analysis
    6. Applies FOV masking to clean boundaries

    Args:
        image_path: Path to retinal fundus image (any common format)

    Returns:
        Binary vessel segmentation as uint8 numpy array (H x W x 3)
        where vessels are white (255, 255, 255) and background is black (0, 0, 0)
    """
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape[:2]

    # === Create retinal FOV mask ===
    # Binary mask distinguishing retinal tissue from black background
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, fov_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    # Smooth FOV boundary
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fov_mask = cv2.morphologyEx(fov_mask, cv2.MORPH_CLOSE, kernel_smooth)

    # Create eroded interior region to prevent boundary artifacts
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fov_interior = cv2.erode(fov_mask, kernel_erode, iterations=3)

    # === Preprocess image ===
    enhanced = preprocess_for_segmentation(arr)

    # === Morphological top-hat for dark tubular structure detection ===
    # Top-hat extracts small dark objects (vessels) by: closing - original
    # where closing = dilation followed by erosion
    kernel_vessel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Use closing instead of opening to extract dark structures
    closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel_vessel)
    tophat = cv2.subtract(closed, enhanced)  # This extracts dark objects
    tophat = cv2.convertScaleAbs(tophat)

    # Normalize tophat response to [0, 255]
    tophat_normalized = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # === Threshold to create vessel mask ===
    # Use Otsu on the tophat response to automatically find threshold
    _, vessel_mask = cv2.threshold(tophat_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # === Connected-component filtering to remove noise ===
    # Label connected components
    labeled_array, num_features = ndimage.label(vessel_mask)

    # Remove small components (likely noise), keep only vessels
    min_size = 3  # minimum pixels per vessel segment
    for i in range(1, num_features + 1):
        component = (labeled_array == i)
        if np.sum(component) < min_size:
            vessel_mask[component] = 0

    # === Apply FOV masking ===
    # Ensure all pixels outside retina are black
    vessel_mask = cv2.bitwise_and(vessel_mask, fov_interior)

    # === Create final RGB output ===
    # Binary white (255, 255, 255) vessels on black (0, 0, 0) background
    segmentation_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    segmentation_rgb[:, :] = [0, 0, 0]  # Black background
    segmentation_rgb[vessel_mask > 127] = [255, 255, 255]  # White vessels

    return segmentation_rgb


def segment_retinal_vessels(image_path: str, output_path: str) -> str | None:
    """
    Complete pipeline for retinal vessel segmentation.

    This is the main entry point that generates a binary vessel segmentation
    visualization saved to disk.

    Args:
        image_path: Path to input retinal fundus image
        output_path: Where to save the output vessel segmentation JPEG

    Returns:
        output_path if successful, None if error occurred
    """
    try:
        print(f"[VesselSegmentation] Segmenting vessels from {image_path}...")

        segmentation_rgb = infer_vessel_segmentation(image_path)

        # Save to disk
        Image.fromarray(segmentation_rgb).save(output_path, "JPEG", quality=92)
        print(f"[VesselSegmentation] Segmentation saved to {output_path}")

        return output_path
    except Exception as e:
        print(f"[VesselSegmentation] Error: {e}")
        import traceback
        traceback.print_exc()
        return None
