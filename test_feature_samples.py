
from pathlib import Path
import numpy as np
from PIL import Image

from src.segmentation import segment_leaf
from src.feature_extractor import extract_features


BASE_DIR = Path("dataset/raw/PlantVillage/color")

sample_files = {
    "Healthy": ("Tomato___healthy", "*RS_HL 0345.JPG"),
    "Early Blight": ("Tomato___Early_blight", "*RS_Erly.B 7844.JPG"),
    "Late Blight": ("Tomato___Late_blight", "*RS_Late.B 6424.JPG")
}


for class_name, (folder_name, pattern) in sample_files.items():

    print("\n" + "=" * 55)
    print(class_name.upper())
    print("=" * 55)

    folder = BASE_DIR / folder_name

    image_path = next(folder.glob(pattern), None)

    if image_path is None:
        print("Sample image not found!")
        continue

    # Load image
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)

    # Segment the leaf
    leaf_mask, segmented_image, segmentation_info = (
        segment_leaf(image_array)
    )

    # Extract features
    features = extract_features(
        image_array,
        leaf_mask
    )

    # Display colour features
    print("\nCOLOUR FEATURES")
    for key, value in features["colour"].items():
        print(f"{key}: {value}")

    # Display shape features
    print("\nSHAPE FEATURES")
    for key, value in features["shape"].items():
        print(f"{key}: {value}")

    # Display texture features
    print("\nTEXTURE FEATURES")
    for key, value in features["texture"].items():
        print(f"{key}: {value}")

    print("\nSEGMENTATION")
    print(segmentation_info)
