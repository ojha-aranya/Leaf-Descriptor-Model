
import pytest

from src.descriptor_generator import generate_descriptor


def create_sample_features():

    return {

        "colour": {
            "dominant_colour": "Green",
            "green_percentage": 99.97,
            "yellow_percentage": 0.01,
            "brown_percentage": 0,
        },

        "shape": {
            "aspect_ratio": 1.05,
            "circularity": 0.3268,
            "solidity": 0.8328,
            "area_pixels": 33528,
            "perimeter_pixels": 1135.4,
        },

        "texture": {
            "contrast": 0.8695,
            "homogeneity": 0.7523,
            "energy": 0.058334,
            "correlation": 0.8658,
        }
    }


def test_descriptor_generation():

    features = create_sample_features()

    descriptor = generate_descriptor(features)

    assert "colour" in descriptor
    assert "shape" in descriptor
    assert "texture" in descriptor
    assert "summary" in descriptor
    assert "disclaimer" in descriptor

    assert descriptor["colour"]["dominant_colour"] == "Green"


def test_descriptor_summary():

    features = create_sample_features()

    descriptor = generate_descriptor(features)

    assert "predominantly green" in descriptor["summary"]


def test_descriptor_disclaimer():

    features = create_sample_features()

    descriptor = generate_descriptor(features)

    assert "diagnosis" in descriptor["disclaimer"]
