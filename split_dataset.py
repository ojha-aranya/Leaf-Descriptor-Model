
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

# --------------------------------------------------
# 1. File locations
# --------------------------------------------------

processed_dir = Path("dataset/processed")

features_file = processed_dir / "tomato_features_deduplicated.csv"
pairs_file = processed_dir / "near_duplicate_report.csv"

# --------------------------------------------------
# 2. Load the feature dataset
# --------------------------------------------------

df = pd.read_csv(features_file)
pairs = pd.read_csv(pairs_file)

print("Total images:", len(df))

# --------------------------------------------------
# 3. Build near-duplicate groups
# --------------------------------------------------

valid_files = set(df["image_filename"])

parent = {filename: filename for filename in valid_files}


def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]


def union(a, b):
    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:
        parent[root_b] = root_a


for _, row in pairs.iterrows():
    image_1 = row["image_1"]
    image_2 = row["image_2"]

    if image_1 in valid_files and image_2 in valid_files:
        union(image_1, image_2)


# Assign one group ID to each image
groups = df["image_filename"].map(find)

print("Unique groups:", groups.nunique())

# --------------------------------------------------
# 4. First split: 80% train+validation, 20% test
# --------------------------------------------------

X = df.drop(columns=["label"])
y = df["label"]

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_val_idx, test_idx = next(
    splitter.split(X, y, groups)
)

train_val = df.iloc[train_val_idx].copy()
test = df.iloc[test_idx].copy()

train_val_groups = groups.iloc[train_val_idx]
test_groups = groups.iloc[test_idx]

# --------------------------------------------------
# 5. Second split: 80% train, 20% validation
# of the remaining 80% (approximately 64/16 overall)
# --------------------------------------------------

second_splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, val_idx = next(
    second_splitter.split(
        train_val,
        train_val["label"],
        train_val_groups
    )
)

train = train_val.iloc[train_idx].copy()
validation = train_val.iloc[val_idx].copy()

# --------------------------------------------------
# 6. Verify group separation
# --------------------------------------------------

train_groups = set(groups.iloc[train.index])
val_groups = set(groups.iloc[validation.index])
test_groups = set(groups.iloc[test.index])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

# --------------------------------------------------
# 7. Save the three datasets
# --------------------------------------------------

train.to_csv(processed_dir / "train.csv", index=False)
validation.to_csv(processed_dir / "validation.csv", index=False)
test.to_csv(processed_dir / "test.csv", index=False)

# --------------------------------------------------
# 8. Display split information
# --------------------------------------------------

print("\n" + "=" * 55)
print("DATASET SPLIT SUMMARY")
print("=" * 55)

for name, data in [
    ("Training", train),
    ("Validation", validation),
    ("Testing", test)
]:
    print(f"\n{name} set")
    print("Images:", len(data))
    print("Percentage:", round(len(data) / len(df) * 100, 2), "%")
    print("Class distribution:")
    print(data["class_name"].value_counts().to_string())

print("\n" + "=" * 55)
print("GROUP LEAKAGE CHECK")
print("=" * 55)

print("Train-validation overlap:", len(train_groups & val_groups))
print("Train-test overlap:", len(train_groups & test_groups))
print("Validation-test overlap:", len(val_groups & test_groups))

print("\nAll three datasets saved successfully.")
