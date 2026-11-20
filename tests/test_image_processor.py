import io
import pytest

from PIL import Image

from src.image_processor import (
    validate_image,
    preprocess_image
)


class MockUpload:

    def __init__(self, data):
        self.data = data

    def getvalue(self):
        return self.data


def create_test_image():

    image = Image.new(
        "RGB",
        (800, 600),
        color="green"
    )

    buffer = io.BytesIO()

    image.save(buffer, format="PNG")

    return buffer.getvalue()


def test_valid_image():

    upload = MockUpload(create_test_image())

    assert validate_image(upload) is True


def test_invalid_image():

    upload = MockUpload(b"This is not an image")

    with pytest.raises(ValueError):
        validate_image(upload)


def test_preprocessing():

    upload = MockUpload(create_test_image())

    image, array, info = preprocess_image(upload)

    assert image.mode == "RGB"

    assert array.shape == (384, 512, 3)

    assert info["channels"] == 3