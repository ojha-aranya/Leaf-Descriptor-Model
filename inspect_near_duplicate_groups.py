
import pandas as pd
from pathlib import Path

# File locations
processed_dir = Path("dataset/processed")

features_file = processed_dir / "tomato_features_deduplicated.csv"
pairs_file = processed_dir / "near_duplicate_report.csv"

# Load data
features = pd.read_csv(features_file)
pairs = pd.read_csv(pairs_file)

# Keep only filenames present in the deduplicated dataset
valid_files = set(features["image_filename"])

parent = {filename: filename for filename in valid_files}


# Find the root parent of an image
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]


# Connect two images into the same group
def union(a, b):
    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:
        parent[root_b] = root_a


# Connect all candidate near-duplicate pairs
for _, row in pairs.iterrows():
    image_1 = row["image_1"]
    image_2 = row["image_2"]

    if image_1 in valid_files and image_2 in valid_files:
        union(image_1, image_2)


# Count images in each connected group
group_sizes = {}

for filename in valid_files:
    root = find(filename)
    group_sizes[root] = group_sizes.get(root, 0) + 1


sizes = pd.Series(list(group_sizes.values()))

print("=" * 55)
print("NEAR-DUPLICATE GROUP INSPECTION")
print("=" * 55)

print("Total images:", len(valid_files))
print("Total connected groups:", len(group_sizes))
print("Largest group:", sizes.max())
print("Groups containing multiple images:", (sizes > 1).sum())

print("\nGroup size distribution:")
print(sizes.value_counts().sort_index().to_string())

print("\nLargest 10 groups:")
print(sizes.sort_values(ascending=False).head(10).to_string(index=False))
