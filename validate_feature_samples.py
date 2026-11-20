
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image

from src.segmentation import segment_leaf
from src.feature_extractor import extract_features


# ---------------------------------
# 1. PROJECT PATHS
# ---------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

BASE_DIR = (
    PROJECT_DIR /
    "dataset" /
    "raw" /
    "PlantVillage" /
    "color"
)

OUTPUT_DIR = PROJECT_DIR / "dataset" / "processed"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = OUTPUT_DIR / "feature_validation_30.csv"


# ---------------------------------
# 2. DEFINE THE THREE CLASSES
# ---------------------------------

classes = {
    "Tomato___healthy": 0,
    "Tomato___Early_blight": 1,
    "Tomato___Late_blight": 2
}

SAMPLES_PER_CLASS = 10

# Fixed seed for reproducibility
random.seed(42)


# ---------------------------------
# 3. SELECT RANDOM IMAGES
# ---------------------------------

selected_images = []

for class_name, label in classes.items():

    folder = BASE_DIR / class_name

    if not folder.exists():
        raise FileNotFoundError(
            f"Class folder not found: {folder}"
        )

    image_files = sorted([
        path for path in folder.iterdir()
        if path.suffix.lower() in [
            ".jpg", ".jpeg", ".png"
        ]
    ])

    if len(image_files) < SAMPLES_PER_CLASS:
        raise ValueError(
            f"Not enough images in {class_name}"
        )

    samples = random.sample(
        image_files,
        SAMPLES_PER_CLASS
    )

    for image_path in samples:

        selected_images.append(
            (image_path, class_name, label)
        )

    print(
        f"{class_name}: "
        f"{len(samples)} images selected"
    )


# ---------------------------------
# 4. EXTRACT FEATURES
# ---------------------------------

all_rows = []
failed_images = []

for index, (image_path, class_name, label) in enumerate(
    selected_images,
    start=1
):

    print(
        f"\nProcessing {index}/30: "
        f"{image_path.name}"
    )

    try:

        # Load image
        with Image.open(image_path) as image:
            image_array = np.array(
                image.convert("RGB")
            )

        # Segment leaf
        leaf_mask, segmented_image, segmentation_info = (
            segment_leaf(image_array)
        )

        # Extract features
        features = extract_features(
            image_array,
            leaf_mask
        )

        # Flatten nested feature dictionary
        row = {
            "image_filename": image_path.name,
            "class_name": class_name,
            "label": label
        }

        for category, values in features.items():

            for feature_name, feature_value in values.items():

                column_name = (
                    f"{category}_{feature_name}"
                )

                row[column_name] = feature_value

        # Add segmentation statistics
        for key, value in segmentation_info.items():

            row[f"segmentation_{key}"] = value

        all_rows.append(row)

        print("SUCCESS")

    except Exception as error:

        failed_images.append({
            "filename": image_path.name,
            "class_name": class_name,
            "error": str(error)
        })

        print("FAILED:", error)


# ---------------------------------
# 5. SAVE FEATURES TO CSV
# ---------------------------------

if all_rows:

    fieldnames = list(all_rows[0].keys())

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_rows)


# ---------------------------------
# 6. DISPLAY FINAL SUMMARY
# ---------------------------------

print("\n" + "=" * 50)
print("FEATURE EXTRACTION SUMMARY")
print("=" * 50)

print("Images selected:", len(selected_images))
print("Successful:", len(all_rows))
print("Failed:", len(failed_images))

for class_name in classes:

    count = sum(
        row["class_name"] == class_name
        for row in all_rows
    )

    print(f"{class_name}: {count} successful images")

print("\nCSV saved at:")
print(OUTPUT_FILE)

if failed_images:

    print("\nFAILED IMAGE DETAILS:")

    for item in failed_images:
        print(item)
