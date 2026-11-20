from pathlib import Path
from PIL import Image, ImageDraw
import pandas as pd

DATASET_PATH = Path("dataset/raw/PlantVillage/color")
REPORT_PATH = Path("dataset/processed/near_duplicate_report.csv")

df = pd.read_csv(REPORT_PATH)

# Select matches belonging to different classes
cross_class = df[df["class_1"] != df["class_2"]]

print("Cross-class candidate pairs:", len(cross_class))

if len(cross_class) == 0:
    print("No cross-class matches found.")
    raise SystemExit

thumb_width = 280
thumb_height = 230
label_height = 65

columns = 2
rows = len(cross_class)

sheet = Image.new(
    "RGB",
    (columns * thumb_width, rows * (thumb_height + label_height)),
    "white"
)

draw = ImageDraw.Draw(sheet)

for index, (_, row) in enumerate(cross_class.iterrows()):

    images = [
        (row["class_1"], row["image_1"]),
        (row["class_2"], row["image_2"])
    ]

    for side, (class_name, filename) in enumerate(images):

        image_path = DATASET_PATH / class_name / filename

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.thumbnail((thumb_width - 10, thumb_height - 10))

            x = side * thumb_width
            y = index * (thumb_height + label_height)

            sheet.paste(img, (x, y))

            draw.text(
                (x + 5, y + thumb_height),
                f"Pair {index + 1}: {class_name}",
                fill="black"
            )

            draw.text(
                (x + 5, y + thumb_height + 20),
                filename[:38],
                fill="black"
            )

        if side == 0:
            draw.text(
                (thumb_width + 5, y + 5),
                f"Hash distance: {row['hash_distance']}",
                fill="red"
            )

output_path = Path(
    "dataset/processed/cross_class_contact_sheet.jpg"
)

sheet.save(output_path)

print("\nContact sheet saved:")
print(output_path.resolve())