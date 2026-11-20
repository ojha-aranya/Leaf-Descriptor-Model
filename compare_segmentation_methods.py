
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

from src.segmentation import segment_leaf


# --------------------------------------------------
# 1. Experimental segmentation method
# --------------------------------------------------

def segment_leaf_hue_guided(image_array):

    if image_array is None or image_array.size == 0:
        raise ValueError("Empty image provided.")

    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("Expected RGB image.")

    if image_array.dtype != np.uint8:
        raise ValueError("Expected uint8 image.")

    height, width = image_array.shape[:2]

    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]

    # Experimental improvement:
    # Exclude highly saturated blue background pixels.
    foreground_seeds = (
        (saturation >= 55) &
        (hue <= 100)
    )

    if np.count_nonzero(foreground_seeds) < 20:
        raise ValueError("No clear foreground seeds found.")

    # Initialize GrabCut mask
    grabcut_mask = np.full(
        (height, width),
        cv2.GC_PR_FGD,
        dtype=np.uint8
    )

    border = max(3, int(min(height, width) * 0.04))

    grabcut_mask[:border, :] = cv2.GC_PR_BGD
    grabcut_mask[-border:, :] = cv2.GC_PR_BGD
    grabcut_mask[:, :border] = cv2.GC_PR_BGD
    grabcut_mask[:, -border:] = cv2.GC_PR_BGD

    definite_border = max(
        2,
        int(min(height, width) * 0.01)
    )

    grabcut_mask[:definite_border, :] = cv2.GC_BGD
    grabcut_mask[-definite_border:, :] = cv2.GC_BGD
    grabcut_mask[:, :definite_border] = cv2.GC_BGD
    grabcut_mask[:, -definite_border:] = cv2.GC_BGD

    # Mark selected leaf-coloured pixels as foreground
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

    # Prepare GrabCut models
    bgr = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )

    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    # Run GrabCut
    cv2.grabCut(
        bgr,
        grabcut_mask,
        None,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_MASK
    )

    initial_mask = np.where(
        (grabcut_mask == cv2.GC_FGD) |
        (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        initial_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No foreground contour found.")

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

    kernel = np.ones((3, 3), dtype=np.uint8)

    leaf_mask = cv2.morphologyEx(
        leaf_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    leaf_pixels = np.count_nonzero(leaf_mask)

    leaf_area_percentage = (
        leaf_pixels / leaf_mask.size
    ) * 100

    if not 1 <= leaf_area_percentage <= 95:
        raise ValueError("Unreliable segmentation area.")

    segmented_image = np.full_like(
        image_array,
        255
    )

    segmented_image[leaf_mask == 255] = (
        image_array[leaf_mask == 255]
    )

    info = {
        "leaf_area_percentage": round(
            leaf_area_percentage,
            2
        )
    }

    return leaf_mask, segmented_image, info


# --------------------------------------------------
# 2. Load misclassified images
# --------------------------------------------------

processed_dir = Path("dataset/processed")

image_dir = (
    Path("dataset")
    / "raw"
    / "PlantVillage"
    / "color"
)

# Load complete validation dataset
df = pd.read_csv(
    processed_dir / "validation.csv"
)

# Select 3 images from each class
selected_groups = []

for class_name in [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight"
]:
    class_images = df[
        df["class_name"] == class_name
    ]

    selected_groups.append(
        class_images.sample(
            n=3,
            random_state=42
        )
    )

selected = pd.concat(
    selected_groups,
    ignore_index=True
)


# --------------------------------------------------
# 3. Compare both segmentation methods
# --------------------------------------------------

fig, axes = plt.subplots(
    len(selected),
    5,
    figsize=(20, 4 * len(selected))
)

for index, (_, row) in enumerate(selected.iterrows()):

    filename = row["image_filename"]
    class_name = row["class_name"]

    image_path = image_dir / class_name / filename

    try:
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)

        # Existing method
        old_mask, old_segmented, old_info = segment_leaf(
            image_array
        )

        # Experimental method
        new_mask, new_segmented, new_info = (
            segment_leaf_hue_guided(image_array)
        )

        axes[index, 0].imshow(image_array)
        axes[index, 0].set_title("Original")

        axes[index, 1].imshow(
            old_mask,
            cmap="gray",
            vmin=0,
            vmax=255
        )
        axes[index, 1].set_title("Current mask")

        axes[index, 2].imshow(
            new_mask,
            cmap="gray",
            vmin=0,
            vmax=255
        )
        axes[index, 2].set_title("Experimental mask")

        axes[index, 3].imshow(old_segmented)
        axes[index, 3].set_title(
            f"Current: {old_info['leaf_area_percentage']}%"
        )

        axes[index, 4].imshow(new_segmented)
        axes[index, 4].set_title(
            f"Experimental: {new_info['leaf_area_percentage']}%"
        )

        print("\nImage:", filename)
        print(
            "Current leaf area:",
            old_info["leaf_area_percentage"], "%"
        )
        print(
            "Experimental leaf area:",
            new_info["leaf_area_percentage"], "%"
        )

    except Exception as error:
        print("\nError processing:", filename)
        print(error)

        for ax in axes[index]:
            ax.axis("off")

    for ax in axes[index]:
        ax.axis("off")


fig.suptitle(
    "Current vs Hue-Guided GrabCut Segmentation",
    fontsize=16
)

plt.tight_layout()

output_path = (
    processed_dir / "segmentation_method_comparison.jpg"
)

plt.savefig(
    output_path,
    dpi=180,
    bbox_inches="tight"
)

plt.show()

print("\nComparison saved to:", output_path)
