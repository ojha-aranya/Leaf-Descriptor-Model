from pathlib import Path
from PIL import Image, ImageDraw

root = Path("dataset/raw/PlantVillage/color")

images = [
    (
        "Tomato___healthy",
        "1479f22a-c0d6-4dac-a752-578f575f276d___GH_HL Leaf 173.5.JPG"
    ),
    (
        "Tomato___Late_blight",
        "cb3bd866-dd84-4f7e-9226-b4a2a2c7f2ae___GH_HL Leaf 173.2.JPG"
    )
]

canvas = Image.new("RGB", (1200, 650), "white")
draw = ImageDraw.Draw(canvas)

for i, (class_name, filename) in enumerate(images):

    path = root / class_name / filename

    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((580, 580))

        x = i * 600

        canvas.paste(img, (x + 10, 50))

        draw.text(
            (x + 10, 15),
            class_name,
            fill="black"
        )

        print(filename)
        print("Original dimensions:", Image.open(path).size)

output = Path("dataset/processed/suspicious_pair_comparison.jpg")
canvas.save(output)

print("\nSaved:", output.resolve())