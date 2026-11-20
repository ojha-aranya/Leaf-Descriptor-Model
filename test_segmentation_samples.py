
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.segmentation import segment_leaf


# Dataset location
BASE_DIR = Path("dataset/raw/PlantVillage/color")

# Three target classes
classes = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight"
]

sample_files = {
    "Tomato___healthy": "*RS_HL 0345.JPG",
    "Tomato___Early_blight": "*RS_Erly.B 7844.JPG",
    "Tomato___Late_blight": "*RS_Late.B 6424.JPG"
}

# Display results
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for row, class_name in enumerate(classes):

    # Find image files
    folder = BASE_DIR / class_name

    image_files = sorted([
        path for path in folder.iterdir()
        if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    if not image_files:
        print(f"No images found in {class_name}")
        continue

    # Select first image
    image_path = next(
    folder.glob(sample_files[class_name]),
    None
    )

    if image_path is None:
        print(f"Sample image not found for {class_name}")
        continue

    # Load image as RGB
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)

    # Apply segmentation
    mask, segmented_image, info = segment_leaf(image_array)
    
    # Check how much detected foreground touches image borders
    border_pixels = np.concatenate([
        mask[0, :],
        mask[-1, :],
        mask[:, 0],
        mask[:, -1]
    ])

    border_foreground = np.count_nonzero(border_pixels)

    print(
        "Foreground pixels touching border:",
        border_foreground
    )

    # Display original image
    axes[row, 0].imshow(image_array)
    axes[row, 0].set_title(f"{class_name}\nOriginal")

    # Display binary mask
    axes[row, 1].imshow(mask, cmap="gray")
    axes[row, 1].set_title("Leaf Mask")

    # Display segmented image
    axes[row, 2].imshow(segmented_image)
    axes[row, 2].set_title("Segmented Leaf")

    print(f"\n{class_name}")
    print("Image:", image_path.name)
    print("Segmentation info:", info)

    for col in range(3):
        axes[row, col].axis("off")

plt.tight_layout()
plt.show()
