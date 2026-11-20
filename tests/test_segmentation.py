import numpy as np
import pytest
import cv2

from src.segmentation import segment_leaf


def create_test_image():

    # White background
    image = np.full(
        (384, 512, 3),
        255,
        dtype=np.uint8
    )

    # Green elliptical leaf
    cv2.ellipse(
        image,
        (256, 192),
        (100, 55),
        0,
        0,
        360,
        (0, 150, 0),
        -1
    )

    return image


def test_segmentation():

    image = create_test_image()

    mask, segmented, info = segment_leaf(image)

    assert mask.shape == (384, 512)

    assert segmented.shape == image.shape

    assert set(np.unique(mask)).issubset({0, 255})

    assert info["leaf_pixels"] > 0

    assert 0 < info["leaf_area_percentage"] < 100


def test_empty_image():

    image = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        segment_leaf(image)