
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

project_dir = Path(".")
processed_dir = project_dir / "dataset" / "processed"

image_dir = (
    project_dir
    / "dataset"
    / "raw"
    / "PlantVillage"
    / "color"
)

csv_path = processed_dir / "misclassified_validation.csv"

df = pd.read_csv(csv_path)

# --------------------------------------------------
# 2. Select 5 mistakes from each disease direction
# --------------------------------------------------

early_to_late = df[
    (df["class_name"] == "Tomato___Early_blight")
    & (df["predicted_class"] == "Late blight")
].head(5)

late_to_early = df[
    (df["class_name"] == "Tomato___Late_blight")
    & (df["predicted_class"] == "Early blight")
].head(5)

selected = pd.concat(
    [early_to_late, late_to_early],
    ignore_index=True
)

# --------------------------------------------------
# 3. Create comparison sheet
# --------------------------------------------------

fig, axes = plt.subplots(2, 5, figsize=(18, 8))

for index, (_, row) in enumerate(selected.iterrows()):

    folder = row["class_name"]
    filename = row["image_filename"]

    image_path = image_dir / folder / filename

    ax = axes[index // 5, index % 5]

    if not image_path.exists():
        ax.set_title("Image not found")
        ax.axis("off")
        print("Missing:", image_path)
        continue

    image = Image.open(image_path).convert("RGB")

    ax.imshow(image)

    ax.set_title(
        f"Actual: {folder.replace('Tomato___', '')}\n"
        f"Predicted: {row['predicted_class']}",
        fontsize=9
    )

    ax.axis("off")

fig.suptitle(
    "Random Forest Misclassified Tomato Images",
    fontsize=16
)

plt.tight_layout()

output_path = processed_dir / "misclassified_comparison.jpg"

plt.savefig(output_path, dpi=200, bbox_inches="tight")

plt.show()

print("\nComparison sheet saved to:", output_path)
