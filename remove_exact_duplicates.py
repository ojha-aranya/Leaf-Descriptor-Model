from pathlib import Path
from collections import defaultdict
import hashlib
import pandas as pd

DATASET_PATH = Path("dataset/raw/PlantVillage/color")
FEATURE_FILE = Path("dataset/processed/tomato_features.csv")

CLASS_NAMES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight"
]


def calculate_hash(path):
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# Step 1: Group images by their exact content
hash_groups = defaultdict(list)

for class_name in CLASS_NAMES:
    folder = DATASET_PATH / class_name

    for path in folder.iterdir():
        if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        image_hash = calculate_hash(path)

        hash_groups[image_hash].append(
            (class_name, path.name)
        )


# Step 2: Keep one image from each group
kept_images = set()
removed_images = []
cross_class_groups = []

for image_hash, images in hash_groups.items():

    images = sorted(images)

    classes = {class_name for class_name, _ in images}

    if len(classes) > 1:
        cross_class_groups.append(images)
        continue

    kept_images.add(images[0])

    for image in images[1:]:
        removed_images.append(image)


# Step 3: Load the original feature dataset
df = pd.read_csv(FEATURE_FILE)

# Step 4: Remove duplicate image records
df["image_key"] = list(zip(
    df["class_name"],
    df["image_filename"]
))

clean_df = df[df["image_key"].isin(kept_images)].copy()

clean_df.drop(columns=["image_key"], inplace=True)

# Step 5: Save cleaned dataset
output_file = Path(
    "dataset/processed/tomato_features_deduplicated.csv"
)

clean_df.to_csv(output_file, index=False)

# Save duplicate report
report = pd.DataFrame(
    removed_images,
    columns=["class_name", "removed_filename"]
)

report.to_csv(
    "dataset/processed/removed_duplicates.csv",
    index=False
)

print("=" * 50)
print("DUPLICATE REMOVAL SUMMARY")
print("=" * 50)

print("Original records:", len(df))
print("Removed duplicate copies:", len(removed_images))
print("Remaining records:", len(clean_df))
print("Cross-class duplicate groups:", len(cross_class_groups))

print("\nClean dataset saved at:")
print(output_file.resolve())

print("\nDuplicate report saved.")