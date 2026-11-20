
import pandas as pd
from pathlib import Path

file_path = Path("dataset/processed/misclassified_validation.csv")

df = pd.read_csv(file_path)

print("=" * 60)
print("MISCLASSIFICATION ANALYSIS")
print("=" * 60)

print("\nTotal incorrect predictions:", len(df))

print("\nActual class vs predicted class:")

summary = pd.crosstab(
    df["class_name"],
    df["predicted_class"],
    rownames=["Actual"],
    colnames=["Predicted"]
)

print(summary)

print("\n" + "=" * 60)
print("SAMPLE MISCLASSIFIED IMAGES")
print("=" * 60)

for (actual, predicted), group in df.groupby(
    ["class_name", "predicted_class"]
):
    print(f"\nActual: {actual}")
    print(f"Predicted: {predicted}")
    print("Number of images:", len(group))

    print("\nSample filenames:")

    for filename in group["image_filename"].head(5):
        print(" -", filename)
