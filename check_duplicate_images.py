from pathlib import Path
from collections import defaultdict
import hashlib

DATASET_PATH = Path(
    "dataset/raw/PlantVillage/color"
)

TARGET_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight"
]


def calculate_hash(image_path):
    sha256 = hashlib.sha256()

    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


hash_groups = defaultdict(list)

total_images = 0

for class_name in TARGET_CLASSES:

    class_path = DATASET_PATH / class_name

    for image_path in class_path.iterdir():

        if image_path.suffix.lower() not in [
            ".jpg", ".jpeg", ".png"
        ]:
            continue

        total_images += 1

        image_hash = calculate_hash(image_path)

        hash_groups[image_hash].append(
            (class_name, image_path.name)
        )


duplicate_groups = {
    image_hash: files
    for image_hash, files in hash_groups.items()
    if len(files) > 1
}

print("=" * 50)
print("EXACT DUPLICATE IMAGE AUDIT")
print("=" * 50)

print("\nTotal images checked:", total_images)
print("Unique image contents:", len(hash_groups))
print("Duplicate groups:", len(duplicate_groups))

duplicate_excess = sum(
    len(files) - 1
    for files in duplicate_groups.values()
)

print("Extra duplicate copies:", duplicate_excess)

for image_hash, files in list(duplicate_groups.items())[:10]:

    print("\nDuplicate group:")

    for class_name, filename in files:
        print(f"  {class_name} -> {filename}")

print("\nAudit completed.")