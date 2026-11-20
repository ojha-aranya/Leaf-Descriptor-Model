from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
import io


# Allowed image formats
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

# Maximum upload size: 10 MB
MAX_FILE_SIZE = 10 * 1024 * 1024

# Standard image dimensions
IMAGE_SIZE = (512, 512)


def validate_image(uploaded_file):

    """
    Validates the uploaded image.

    Checks:
    1. File size
    2. Image format
    3. Image integrity
    """

    if uploaded_file is None:
        raise ValueError("No image uploaded.")

    image_bytes = uploaded_file.getvalue()

    if len(image_bytes) == 0:
        raise ValueError("Uploaded file is empty.")

    if len(image_bytes) > MAX_FILE_SIZE:
        raise ValueError("Image size exceeds 10 MB.")

    try:

        image = Image.open(io.BytesIO(image_bytes))

        if image.format not in ALLOWED_FORMATS:
            raise ValueError("Unsupported image format.")

        image.verify()

        return True

    except (UnidentifiedImageError, OSError, SyntaxError):
        raise ValueError("Invalid or corrupted image file.")


def preprocess_image(uploaded_file):

    """
    Performs image preprocessing.

    Returns:
    1. Processed PIL image
    2. NumPy image array
    3. Image information
    """

    validate_image(uploaded_file)

    image_bytes = uploaded_file.getvalue()

    image = Image.open(io.BytesIO(image_bytes))

    # Correct image orientation using EXIF metadata
    image = ImageOps.exif_transpose(image)

    # Convert image to RGB
    image = image.convert("RGB")

    # Store original dimensions
    original_width, original_height = image.size

    # Resize while maintaining aspect ratio
    image.thumbnail(IMAGE_SIZE, Image.Resampling.LANCZOS)

    # Convert to NumPy array
    image_array = np.array(image)

    # Collect image information
    image_info = {
        "original_width": original_width,
        "original_height": original_height,
        "processed_width": image.width,
        "processed_height": image.height,
        "channels": image_array.shape[2],
        "data_type": str(image_array.dtype),
        "file_size_kb": round(
            len(image_bytes) / 1024, 2
        )
    }

    return image, image_array, image_info