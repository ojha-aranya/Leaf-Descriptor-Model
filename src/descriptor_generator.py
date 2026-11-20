
def generate_descriptor(features):
    """
    Converts numerical leaf features into
    a structured, human-readable descriptor.

    Input:
        features: Dictionary from feature_extractor.py

    Output:
        Structured descriptor dictionary
    """

    colour = features["colour"]
    shape = features["shape"]
    texture = features["texture"]

    # --------------------------------
    # 1. COLOUR DESCRIPTION
    # --------------------------------

    dominant_colour = colour["dominant_colour"]

    green_percentage = colour["green_percentage"]
    yellow_percentage = colour["yellow_percentage"]
    brown_percentage = colour["brown_percentage"]

    if green_percentage >= 70:
        colour_description = (
            "The detected region is predominantly green."
        )
    elif yellow_percentage >= 30:
        colour_description = (
            "A substantial yellow colour component was detected."
        )
    elif brown_percentage >= 30:
        colour_description = (
            "A substantial brown colour component was detected."
        )
    else:
        colour_description = (
            "The detected region contains a mixed colour distribution."
        )

    # --------------------------------
    # 2. SHAPE DESCRIPTION
    # --------------------------------

    aspect_ratio = shape["aspect_ratio"]
    circularity = shape["circularity"]
    solidity = shape["solidity"]

    if aspect_ratio > 1.2:
        shape_description = "The bounding box is wider than it is tall."
    elif aspect_ratio < 0.8:
        shape_description = "The bounding box is taller than it is wide."
    else:
        shape_description = (
            "The bounding box has approximately equal width and height."
        )

    if circularity >= 0.7:
        boundary_description = "The outline is relatively circular."
    elif circularity >= 0.4:
        boundary_description = "The outline has moderate irregularity."
    else:
        boundary_description = "The outline is highly non-circular."

    if solidity >= 0.9:
        solidity_description = "The outline has relatively few concavities."
    elif solidity >= 0.75:
        solidity_description = "The outline has noticeable concavities."
    else:
        solidity_description = "The outline has substantial concavities."

    # --------------------------------
    # 3. TEXTURE DESCRIPTION
    # --------------------------------

    contrast = texture["contrast"]
    homogeneity = texture["homogeneity"]
    energy = texture["energy"]
    correlation = texture["correlation"]

    texture_description = (
        "Texture measurements were successfully extracted "
        "using a 16-level horizontal GLCM."
    )

    # --------------------------------
    # 4. BUILD STRUCTURED DESCRIPTOR
    # --------------------------------

    descriptor = {

        "colour": {
            "dominant_colour": dominant_colour,
            "green_percentage": green_percentage,
            "yellow_percentage": yellow_percentage,
            "brown_percentage": brown_percentage,
            "description": colour_description
        },

        "shape": {
            "aspect_ratio": aspect_ratio,
            "circularity": circularity,
            "solidity": solidity,
            "area_pixels": shape["area_pixels"],
            "perimeter_pixels": shape["perimeter_pixels"],
            "description": shape_description,
            "boundary_description": boundary_description,
            "solidity_description": solidity_description
        },

        "texture": {
            "contrast": contrast,
            "homogeneity": homogeneity,
            "energy": energy,
            "correlation": correlation,
            "description": texture_description
        },

        "summary": (
            f"The analysed leaf region is predominantly "
            f"{dominant_colour.lower()}. "
            f"{shape_description} "
            f"{boundary_description} "
            f"{solidity_description}"
        ),

        "disclaimer": (
            "This descriptor represents measurable visual "
            "characteristics. It is not a scientifically "
            "validated plant disease diagnosis."
        )
    }

    return descriptor
