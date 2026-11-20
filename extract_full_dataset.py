
import csv
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

OUTPUT_DIR = (
    PROJECT_DIR /
    "dataset" /
    "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = OUTPUT_DIR / "tomato_features.csv"
FAILED_FILE = OUTPUT_DIR / "failed_images.csv"


# ---------------------------------
# 2. CLASS DEFINITIONS
# ---------------------------------

classes = {
    "Tomato___healthy": 0,
    "Tomato___Early_blight": 1,
    "Tomato___Late_blight": 2
}

valid_extensions = {
    ".jpg", ".jpeg", ".png"
}


# ---------------------------------
# 3. COLLECT ALL IMAGE PATHS
# ---------------------------------

all_images = []

for class_name, label in classes.items():

    folder = BASE_DIR / class_name

    if not folder.exists():
        raise FileNotFoundError(
            f"Class folder not found: {folder}"
        )

    image_files = sorted([
        path for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in valid_extensions
    ])

    print(
        f"{class_name}: {len(image_files)} images"
    )

    for image_path in image_files:
        all_images.append(
            (image_path, class_name, label)
        )


total_images = len(all_images)

print("\nTotal images to process:", total_images)


# ---------------------------------
# 4. PROCESS AND SAVE FEATURES
# ---------------------------------

successful = 0
failed = 0

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as feature_file, open(
    FAILED_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as failed_file:

    feature_writer = None

    failed_writer = csv.DictWriter(
        failed_file,
        fieldnames=[
            "filename",
            "class_name",
            "error"
        ]
    )

    failed_writer.writeheader()

    for index, (image_path, class_name, label) in enumerate(
        all_images,
        start=1
    ):

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

            # Extract numerical features
            features = extract_features(
                image_array,
                leaf_mask
            )

            # Create one CSV row
            row = {
                "image_filename": image_path.name,
                "class_name": class_name,
                "label": label
            }

            # Flatten feature dictionary
            for category, values in features.items():

                for feature_name, feature_value in values.items():

                    column_name = (
                        f"{category}_{feature_name}"
                    )

                    row[column_name] = feature_value

            # Add segmentation statistics
            for key, value in segmentation_info.items():

                row[f"segmentation_{key}"] = value

            # Create CSV header from first successful image
            if feature_writer is None:

                feature_writer = csv.DictWriter(
                    feature_file,
                    fieldnames=list(row.keys())
                )

                feature_writer.writeheader()

            feature_writer.writerow(row)

            # Save progress immediately
            feature_file.flush()

            successful += 1

        except Exception as error:

            failed_writer.writerow({
                "filename": image_path.name,
                "class_name": class_name,
                "error": str(error)
            })

            failed_file.flush()

            failed += 1

        # Display progress every 25 images
        if index % 25 == 0 or index == total_images:

            print(
                f"Progress: {index}/{total_images} | "
                f"Successful: {successful} | "
                f"Failed: {failed}"
            )


# ---------------------------------
# 5. FINAL SUMMARY
# ---------------------------------

print("\n" + "=" * 55)
print("FULL DATASET EXTRACTION COMPLETED")
print("=" * 55)

print("Total images:", total_images)
print("Successful:", successful)
print("Failed:", failed)

print("\nFeature dataset:")
print(OUTPUT_FILE)

print("\nFailed image report:")
print(FAILED_FILE)
