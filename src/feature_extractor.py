
import cv2
import numpy as np


def extract_features(image_array, leaf_mask):
    """
    Extracts colour, shape and texture features
    from a segmented leaf.

    Input:
        image_array: Original RGB image (uint8 NumPy array)
        leaf_mask: Binary segmentation mask

    Returns:
        Dictionary containing extracted features
    """

    # ---------------------------------
    # INPUT VALIDATION
    # ---------------------------------

    if image_array is None or image_array.size == 0:
        raise ValueError("Empty image provided.")

    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("Expected an RGB image.")

    if image_array.dtype != np.uint8:
        raise ValueError("Expected an 8-bit RGB image.")

    if leaf_mask is None:
        raise ValueError("Leaf mask is missing.")

    if leaf_mask.shape != image_array.shape[:2]:
        raise ValueError("Image and mask dimensions do not match.")

    mask = (leaf_mask > 0).astype(np.uint8)

    if np.count_nonzero(mask) == 0:
        raise ValueError("No leaf pixels found.")

    # ---------------------------------
    # 1. COLOUR FEATURE EXTRACTION
    # ---------------------------------

    rgb_pixels = image_array[mask == 1]

    mean_rgb = np.mean(rgb_pixels, axis=0)

    hsv_image = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2HSV
    )

    hsv_pixels = hsv_image[mask == 1]

    mean_hsv = np.mean(hsv_pixels, axis=0)

    hue = hsv_pixels[:, 0]
    saturation = hsv_pixels[:, 1]
    value = hsv_pixels[:, 2]

    # Brown: darker, sufficiently saturated pixels
    brown_mask = (
        (hue <= 25) &
        (saturation >= 45) &
        (value < 180)
    )

    # Yellow: brighter yellow pixels
    yellow_mask = (
        (hue > 25) &
        (hue <= 35) &
        (saturation >= 40) &
        (value >= 70)
    )

    # Green: green hue range
    green_mask = (
        (hue > 35) &
        (hue <= 95) &
        (saturation >= 30) &
        (value >= 30)
    )

    # Dark pixels are measured separately.
    # They may include shadows or damaged tissue.
    dark_mask = value < 60

    total_leaf_pixels = len(hsv_pixels)

    green_pixels = np.count_nonzero(green_mask)
    yellow_pixels = np.count_nonzero(yellow_mask)
    brown_pixels = np.count_nonzero(brown_mask)
    dark_pixels = np.count_nonzero(dark_mask)

    green_percentage = (
        green_pixels / total_leaf_pixels
    ) * 100

    yellow_percentage = (
        yellow_pixels / total_leaf_pixels
    ) * 100

    brown_percentage = (
        brown_pixels / total_leaf_pixels
    ) * 100

    dark_percentage = (
        dark_pixels / total_leaf_pixels
    ) * 100

    # Determine dominant colour using pixel counts
    colour_counts = {
        "Green": green_pixels,
        "Yellow": yellow_pixels,
        "Brown": brown_pixels
    }

    if max(colour_counts.values()) == 0:
        dominant_colour = "Other"
    else:
        dominant_colour = max(
            colour_counts,
            key=colour_counts.get
        )

    # ---------------------------------
    # 2. SHAPE FEATURE EXTRACTION
    # ---------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No leaf contour found.")

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(largest_contour)

    perimeter = cv2.arcLength(
        largest_contour,
        True
    )

    x, y, width, height = cv2.boundingRect(
        largest_contour
    )

    aspect_ratio = (
        width / height if height > 0 else 0
    )

    circularity = (
        (4 * np.pi * area) / (perimeter ** 2)
        if perimeter > 0 else 0
    )

    hull = cv2.convexHull(largest_contour)

    hull_area = cv2.contourArea(hull)

    solidity = (
        area / hull_area if hull_area > 0 else 0
    )

    # ---------------------------------
    # 3. TEXTURE FEATURE EXTRACTION
    # ---------------------------------

    gray_image = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    # Reduce grayscale levels to 16
    quantized = (
        gray_image // 16
    ).astype(np.uint8)

    # Compare horizontally adjacent pixels
    valid_pairs = (
        (mask[:, :-1] == 1) &
        (mask[:, 1:] == 1)
    )

    first = quantized[:, :-1][valid_pairs]
    second = quantized[:, 1:][valid_pairs]

    if len(first) == 0:
        raise ValueError(
            "Insufficient pixels for texture analysis."
        )

    # Build a 16 x 16 co-occurrence matrix
    glcm = np.zeros(
        (16, 16),
        dtype=np.float64
    )

    np.add.at(glcm, (first, second), 1)

    # Make matrix symmetric
    glcm = glcm + glcm.T

    # Normalise matrix
    glcm = glcm / glcm.sum()

    levels = np.arange(16)

    i, j = np.meshgrid(
        levels,
        levels,
        indexing="ij"
    )

    # Contrast
    contrast = np.sum(
        glcm * ((i - j) ** 2)
    )

    # Homogeneity
    homogeneity = np.sum(
        glcm / (1 + np.abs(i - j))
    )

    # Energy
    energy = np.sum(glcm ** 2)

    # Correlation
    mean_i = np.sum(glcm * i)
    mean_j = np.sum(glcm * j)

    std_i = np.sqrt(
        np.sum(glcm * ((i - mean_i) ** 2))
    )

    std_j = np.sqrt(
        np.sum(glcm * ((j - mean_j) ** 2))
    )

    if std_i > 0 and std_j > 0:

        correlation = np.sum(
            glcm * (i - mean_i) * (j - mean_j)
        ) / (std_i * std_j)

    else:
        correlation = 0.0

    # ---------------------------------
    # 4. COMBINE ALL FEATURES
    # ---------------------------------

    features = {

        "colour": {
            "mean_red": round(float(mean_rgb[0]), 2),
            "mean_green": round(float(mean_rgb[1]), 2),
            "mean_blue": round(float(mean_rgb[2]), 2),

            "mean_hue": round(float(mean_hsv[0]), 2),
            "mean_saturation": round(float(mean_hsv[1]), 2),
            "mean_value": round(float(mean_hsv[2]), 2),

            "dominant_colour": dominant_colour,

            "green_percentage": round(
                float(green_percentage), 2
            ),

            "yellow_percentage": round(
                float(yellow_percentage), 2
            ),

            "brown_percentage": round(
                float(brown_percentage), 2
            ),

            "dark_percentage": round(
                float(dark_percentage), 2
            )
        },

        "shape": {
            "area_pixels": round(float(area), 2),
            "perimeter_pixels": round(float(perimeter), 2),
            "aspect_ratio": round(float(aspect_ratio), 3),
            "circularity": round(float(circularity), 4),
            "solidity": round(float(solidity), 4),
            "bounding_width": int(width),
            "bounding_height": int(height)
        },

        "texture": {
            "contrast": round(float(contrast), 4),
            "homogeneity": round(float(homogeneity), 4),
            "energy": round(float(energy), 6),
            "correlation": round(float(correlation), 4)
        },

        "analysis_info": {
            "analysed_pixels": int(total_leaf_pixels),
            "texture_method": "16-level horizontal GLCM"
        }
    }

    return features
