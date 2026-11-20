from pathlib import Path
from PIL import Image
import imagehash
import pandas as pd
from itertools import combinations

DATASET_PATH = Path("dataset/raw/PlantVillage/color")

CLASS_NAMES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight"
]

# Maximum Hamming distance for a potential match
HASH_THRESHOLD = 8

image_hashes = []

print("Calculating perceptual hashes...")

for class_name in CLASS_NAMES:

    folder = DATASET_PATH / class_name

    for image_path in folder.iterdir():

        if image_path.suffix.lower() not in [
            ".jpg", ".jpeg", ".png"
        ]:
            continue

        try:
            with Image.open(image_path) as img:
                img_hash = imagehash.phash(img.convert("RGB"))

            image_hashes.append({
                "filename": image_path.name,
                "class_name": class_name,
                "hash": img_hash
            })

        except Exception as error:
            print("Could not process:", image_path.name, error)


print("\nTotal images hashed:", len(image_hashes))
print("Comparing image pairs...")

matches = []

for image1, image2 in combinations(image_hashes, 2):

    distance = image1["hash"] - image2["hash"]

    if distance <= HASH_THRESHOLD:

        matches.append({
            "image_1": image1["filename"],
            "class_1": image1["class_name"],
            "image_2": image2["filename"],
            "class_2": image2["class_name"],
            "hash_distance": distance
        })


results = pd.DataFrame(matches)

output_path = Path(
    "dataset/processed/near_duplicate_report.csv"
)

results.to_csv(output_path, index=False)

print("\n" + "=" * 50)
print("NEAR-DUPLICATE AUDIT")
print("=" * 50)

print("Total images:", len(image_hashes))
print("Potential near-duplicate pairs:", len(results))

if len(results) > 0:

    print("\nPairs by class combination:")

    print(
        results.groupby(["class_1", "class_2"])
        .size()
    )

    print("\nClosest matches:")
    print(
        results.sort_values("hash_distance")
        .head(20)
        .to_string(index=False)
    )

else:
    print("\nNo potential near-duplicates found.")

print("\nReport saved at:")
print(output_path.resolve())