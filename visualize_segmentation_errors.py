
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

from src.segmentation import segment_leaf

# --------------------------------------------------
# 1. File locations
# --------------------------------------------------

project_dir = Path(".")
processed_dir = project_dir / "dataset" / "processed"

image_dir = (
    project_dir
    / "dataset"
    / "raw"
    / "PlantVillage"
    / "color"
)

csv_path = processed_dir / "misclassified_validation.csv"

df = pd.read_csv(csv_path)

# --------------------------------------------------
# 2. Select three images from each error category
# --------------------------------------------------

early_to_late = df[
    (df["class_name"] == "Tomato___Early_blight")
    & (df["predicted_class"] == "Late blight")
].head(3)

late_to_early = df[
    (df["class_name"] == "Tomato___Late_blight")
    & (df["predicted_class"] == "Early blight")
].head(3)

selected = pd.concat(
    [early_to_late, late_to_early],
    ignore_index=True
)

# --------------------------------------------------
# 3. Create visualization
# --------------------------------------------------

fig, axes = plt.subplots(
    len(selected),
    3,
    figsize=(13, 4 * len(selected))
)

for index, (_, row) in enumerate(selected.iterrows()):

    folder = row["class_name"]
    filename = row["image_filename"]

    image_path = image_dir / folder / filename

    if not image_path.exists():
        print("Image not found:", image_path)
        continue

    # Load original RGB image
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)

    # Run existing segmentation
    leaf_mask, segmented_image, info = segment_leaf(
        image_array
    )

    # Display original
    axes[index, 0].imshow(image_array)
    axes[index, 0].set_title("Original image")

    # Display binary mask
    axes[index, 1].imshow(
        leaf_mask,
        cmap="gray",
        vmin=0,
        vmax=255
    )
    axes[index, 1].set_title("Binary segmentation mask")

    # Display segmented image
    axes[index, 2].imshow(segmented_image)
    axes[index, 2].set_title(
        f"Segmented leaf\n"
        f"Leaf area: {info['leaf_area_percentage']}%"
    )

    for ax in axes[index]:
        ax.axis("off")

    print(
        f"\n{filename}: "
        f"Leaf area = {info['leaf_area_percentage']}%"
    )

fig.suptitle(
    "Segmentation Inspection - Misclassified Images",
    fontsize=16
)

plt.tight_layout()

output_path = (
    processed_dir / "segmentation_error_inspection.jpg"
)

plt.savefig(
    output_path,
    dpi=180,
    bbox_inches="tight"
)

plt.show()

print("\nSaved visualization:", output_path)
