
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from src.segmentation import segment_leaf
from compare_segmentation_methods import segment_leaf_hue_guided

image_dir = Path("dataset/raw/PlantVillage/color")
output_dir = Path("dataset/processed")

images = [
    (
        "Tomato___Early_blight",
        "a6041f09-68f2-4865-ad78-81018617d2c4___RS_Erly.B 6450.JPG"
    ),
    (
        "Tomato___Late_blight",
        "06a7cfd9-0992-4020-8a11-79249d657c0c___GHLB2 Leaf 131.JPG"
    )
]

fig, axes = plt.subplots(2, 5, figsize=(20, 10))

for row, (class_name, filename) in enumerate(images):

    image_path = image_dir / class_name / filename

    image = np.array(
        Image.open(image_path).convert("RGB")
    )

    old_mask, old_segmented, old_info = segment_leaf(image)

    new_mask, new_segmented, new_info = (
        segment_leaf_hue_guided(image)
    )

    axes[row, 0].imshow(image)
    axes[row, 0].set_title("Original")

    axes[row, 1].imshow(old_mask, cmap="gray")
    axes[row, 1].set_title("Current mask")

    axes[row, 2].imshow(new_mask, cmap="gray")
    axes[row, 2].set_title("Experimental mask")

    axes[row, 3].imshow(old_segmented)
    axes[row, 3].set_title(
        f"Current: {old_info['leaf_area_percentage']}%"
    )

    axes[row, 4].imshow(new_segmented)
    axes[row, 4].set_title(
        f"Experimental: {new_info['leaf_area_percentage']}%"
    )

    for ax in axes[row]:
        ax.axis("off")

    print("\n", filename)
    print("Current:", old_info["leaf_area_percentage"])
    print("Experimental:", new_info["leaf_area_percentage"])

plt.tight_layout()

plt.savefig(
    output_dir / "detailed_segmentation_comparison.jpg",
    dpi=200,
    bbox_inches="tight"
)

plt.show()
