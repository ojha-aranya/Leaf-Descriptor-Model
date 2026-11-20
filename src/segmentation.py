
import cv2
import numpy as np


def segment_leaf(image_array):
    """
    Detects the complete leaf using GrabCut segmentation.

    Input:
        image_array: RGB NumPy image

    Returns:
        leaf_mask: Binary mask (leaf = 255, background = 0)
        segmented_image: Leaf with white background
        segmentation_info: Segmentation statistics
    """

    # Step 1: Validate the input
    if image_array is None or image_array.size == 0:
        raise ValueError("Empty image provided.")

    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("Expected an RGB image.")

    if image_array.dtype != np.uint8:
        raise ValueError("Expected an 8-bit RGB image.")

    height, width = image_array.shape[:2]

    if height < 20 or width < 20:
        raise ValueError("Image is too small for segmentation.")

    # Step 2: Convert RGB image to HSV
    hsv_image = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2HSV
    )

    saturation = hsv_image[:, :, 1]

    # Step 3: Check whether the image has enough
    # coloured pixels to identify a possible leaf
    foreground_seeds = saturation >= 55

    if np.count_nonzero(foreground_seeds) < 20:
        raise ValueError(
            "No clear leaf region detected."
        )

    # Step 4: Initialise the GrabCut mask
    # 0 = definite background
    # 1 = probable background
    # 2 = probable foreground
    # 3 = definite foreground

    grabcut_mask = np.full(
        (height, width),
        cv2.GC_PR_FGD,
        dtype=np.uint8
    )

    # Step 5: Mark the outer border as probable background
    border = max(3, int(min(height, width) * 0.04))

    grabcut_mask[:border, :] = cv2.GC_PR_BGD
    grabcut_mask[-border:, :] = cv2.GC_PR_BGD
    grabcut_mask[:, :border] = cv2.GC_PR_BGD
    grabcut_mask[:, -border:] = cv2.GC_PR_BGD

    # Step 6: Mark the inner border as definite background
    definite_border = max(2, int(min(height, width) * 0.01))

    grabcut_mask[:definite_border, :] = cv2.GC_BGD
    grabcut_mask[-definite_border:, :] = cv2.GC_BGD
    grabcut_mask[:, :definite_border] = cv2.GC_BGD
    grabcut_mask[:, -definite_border:] = cv2.GC_BGD

    # Step 7: Mark sufficiently saturated interior pixels
    # as definite foreground
    interior = np.zeros(
        (height, width),
        dtype=bool
    )

    interior[
        border:height-border,
        border:width-border
    ] = True

    grabcut_mask[
        foreground_seeds & interior
    ] = cv2.GC_FGD

    # Step 8: Convert RGB to BGR for OpenCV GrabCut
    bgr_image = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )

    # Step 9: Prepare GrabCut models
    background_model = np.zeros(
        (1, 65),
        dtype=np.float64
    )

    foreground_model = np.zeros(
        (1, 65),
        dtype=np.float64
    )

    # Step 10: Run GrabCut
    cv2.grabCut(
        bgr_image,
        grabcut_mask,
        None,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_MASK
    )

    # Step 11: Extract foreground pixels
    initial_leaf_mask = np.where(
        (grabcut_mask == cv2.GC_FGD) |
        (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    # Step 12: Remove small isolated regions
    contours, _ = cv2.findContours(
        initial_leaf_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError(
            "No leaf region could be segmented."
        )

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    leaf_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    cv2.drawContours(
        leaf_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # Step 13: Close small gaps in the leaf mask
    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    leaf_mask = cv2.morphologyEx(
        leaf_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Step 14: Validate the resulting mask
    total_pixels = leaf_mask.size
    leaf_pixels = np.count_nonzero(leaf_mask)

    leaf_area_percentage = (
        leaf_pixels / total_pixels
    ) * 100

    if leaf_area_percentage < 1 or leaf_area_percentage > 95:
        raise ValueError(
            "Segmentation produced an unreliable leaf mask."
        )

    # Step 15: Apply the mask to the original RGB image
    segmented_image = np.full_like(
        image_array,
        255
    )

    segmented_image[leaf_mask == 255] = (
        image_array[leaf_mask == 255]
    )

    # Step 16: Calculate segmentation statistics
    segmentation_info = {
        "leaf_pixels": int(leaf_pixels),

        "background_pixels": int(
            total_pixels - leaf_pixels
        ),

        "leaf_area_percentage": round(
            leaf_area_percentage,
            2
        )
    }

    return leaf_mask, segmented_image, segmentation_info
