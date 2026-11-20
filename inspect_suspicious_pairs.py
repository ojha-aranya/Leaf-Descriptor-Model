
from pathlib import Path
from PIL import Image, ImageDraw
import pandas as pd

root = Path("dataset/raw/PlantVillage/color")

df = pd.read_csv(
    "dataset/processed/near_duplicate_report.csv"
)

targets = [
    "GH_HL Leaf 173.5.JPG",
    "GH_HL Leaf 173.2.JPG",
    "GH_HL Leaf 485.JPG",
    "GH_HL Leaf 170.1.JPG",
    "GH_HL Leaf 170.JPG"
]

files = {}

# Find filenames containing the target text
for _, row in df.iterrows():

    for filename, class_name in [
        (row["image_1"], row["class_1"]),
        (row["image_2"], row["class_2"])
    ]:

        if any(target in filename for target in targets):
            files[(class_name, filename)] = True

print("Matching images found:", len(files))

images = []

for class_name, filename in files:

    path = root / class_name / filename

    if not path.exists():
        print("File not found:", path)
        continue

    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((380, 380))
        images.append((filename, class_name, img.copy()))

if not images:
    raise RuntimeError("No images found. Check filenames and paths.")

# Arrange images in two columns
columns = 2
rows = (len(images) + 1) // 2

card_width = 420
card_height = 440

sheet = Image.new(
    "RGB",
    (columns * card_width, rows * card_height),
    "white"
)

draw = ImageDraw.Draw(sheet)

for i, (filename, class_name, img) in enumerate(images):

    x = (i % columns) * card_width
    y = (i // columns) * card_height

    sheet.paste(img, (x + 10, y + 35))

    draw.text(
        (x + 10, y + 5),
        class_name,
        fill="black"
    )

    draw.text(
        (x + 10, y + 420),
        filename[:48],
        fill="black"
    )

output = Path(
    "dataset/processed/suspicious_images.jpg"
)

sheet.save(output)

print("\nImages included:", len(images))
print("Saved:", output.resolve())
