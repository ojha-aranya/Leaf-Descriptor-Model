
import numpy as np
import pytest
import cv2

from src.segmentation import segment_leaf
from src.feature_extractor import extract_features


def create_test_image():

    image = np.full(
        (384, 512, 3),
        255,
        dtype=np.uint8
    )

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


def test_feature_extraction():

    image = create_test_image()

    mask, segmented, info = segment_leaf(image)

    features = extract_features(image, mask)

    assert "colour" in features
    assert "shape" in features
    assert "texture" in features

    assert features["colour"]["dominant_colour"] == "Green"

    assert features["shape"]["area_pixels"] > 0

    assert features["shape"]["perimeter_pixels"] > 0

    assert features["texture"]["contrast"] >= 0

    assert 0 <= features["shape"]["solidity"] <= 1


def test_invalid_mask():

    image = create_test_image()

    empty_mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8
    )

    with pytest.raises(ValueError):
        extract_features(image, empty_mask)


def test_wrong_mask_dimensions():

    image = create_test_image()

    wrong_mask = np.zeros(
        (100, 100),
        dtype=np.uint8
    )

    with pytest.raises(ValueError):
        extract_features(image, wrong_mask)
