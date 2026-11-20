
from pathlib import Path
import pandas as pd

report_path = Path(
    "dataset/processed/near_duplicate_report.csv"
)

output_path = Path(
    "dataset/processed/cross_class_manual_review.csv"
)

# Load near-duplicate report
df = pd.read_csv(report_path)

# Keep only cross-class pairs
review_df = df[
    df["class_1"] != df["class_2"]
].copy()

# Add manual review columns
review_df["review_status"] = "Pending"

review_df["review_decision"] = ""

review_df["review_notes"] = ""

# Add image paths for convenient inspection
review_df["image_1_path"] = review_df.apply(
    lambda row:
        f"dataset/raw/PlantVillage/color/"
        f"{row['class_1']}/{row['image_1']}",
    axis=1
)

review_df["image_2_path"] = review_df.apply(
    lambda row:
        f"dataset/raw/PlantVillage/color/"
        f"{row['class_2']}/{row['image_2']}",
    axis=1
)

# Save review register
review_df.to_csv(output_path, index=False)

print("=" * 50)
print("CROSS-CLASS MANUAL REVIEW REGISTER")
print("=" * 50)

print("Pairs requiring review:", len(review_df))

print("\nReview status:")
print(review_df["review_status"].value_counts())

print("\nSaved at:")
print(output_path.resolve())
