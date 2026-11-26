import pickle
import pandas as pd
from pathlib import Path
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Leaf Descriptor Model",
    page_icon="🌿",
    layout="wide"
)


from src.image_processor import preprocess_image
from src.segmentation import segment_leaf
from src.feature_extractor import extract_features
from src.descriptor_generator import generate_descriptor


# Load trained classification model
@st.cache_resource
def load_model():

    model_path = (
        Path("dataset")
        / "processed"
        / "tomato_random_forest.pkl"
    )

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model


model = load_model()

def prepare_model_input(features):

    model_features = {
        "colour_mean_red":
            features["colour"]["mean_red"],

        "colour_mean_green":
            features["colour"]["mean_green"],

        "colour_mean_blue":
            features["colour"]["mean_blue"],

        "colour_mean_hue":
            features["colour"]["mean_hue"],

        "colour_mean_saturation":
            features["colour"]["mean_saturation"],

        "colour_mean_value":
            features["colour"]["mean_value"],

        "colour_dominant_colour":
            features["colour"]["dominant_colour"],

        "colour_green_percentage":
            features["colour"]["green_percentage"],

        "colour_yellow_percentage":
            features["colour"]["yellow_percentage"],

        "colour_brown_percentage":
            features["colour"]["brown_percentage"],

        "colour_dark_percentage":
            features["colour"]["dark_percentage"],

        "shape_area_pixels":
            features["shape"]["area_pixels"],

        "shape_perimeter_pixels":
            features["shape"]["perimeter_pixels"],

        "shape_aspect_ratio":
            features["shape"]["aspect_ratio"],

        "shape_circularity":
            features["shape"]["circularity"],

        "shape_solidity":
            features["shape"]["solidity"],

        "shape_bounding_width":
            features["shape"]["bounding_width"],

        "shape_bounding_height":
            features["shape"]["bounding_height"],

        "texture_contrast":
            features["texture"]["contrast"],

        "texture_homogeneity":
            features["texture"]["homogeneity"],

        "texture_energy":
            features["texture"]["energy"],

        "texture_correlation":
            features["texture"]["correlation"]
    }

    return pd.DataFrame([model_features])



# Application heading
st.title("🌿 Leaf Descriptor Model")

st.subheader(
    "Computer Vision-Based Leaf Analysis "
    "and Health Screening"
)

st.write(
    "Upload a plant leaf image to begin "
    "visual analysis and preprocessing."
)


st.divider()


# Image uploader
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded_file is not None:

    try:

        # Process uploaded image
        processed_image, image_array, image_info = (
            preprocess_image(uploaded_file)
        )

        st.success("Image uploaded and processed successfully!")

        # Display image
        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Processed Leaf Image")

            st.image(
                processed_image,
                caption="Preprocessed RGB Image",
                use_container_width=True
            )

        with col2:

            st.subheader("Image Information")

            st.write(
                f"Original Width: {image_info['original_width']} px"
            )

            st.write(
                f"Original Height: {image_info['original_height']} px"
            )

            st.write(
                f"Processed Width: {image_info['processed_width']} px"
            )

            st.write(
                f"Processed Height: {image_info['processed_height']} px"
            )

            st.write(
                f"Channels: {image_info['channels']}"
            )

            st.write(
                f"Data Type: {image_info['data_type']}"
            )

            st.write(
                f"File Size: {image_info['file_size_kb']} KB"
            )

        st.divider()

        st.subheader("NumPy Image Representation")

        st.write("Image array shape:", image_array.shape)

        st.write("Image array data type:", image_array.dtype)

        st.write("Sample pixel values:")

        st.code(str(image_array[0, 0]))
        
        st.divider()

        st.header("Leaf Segmentation")

        # Perform segmentation
        leaf_mask, segmented_image, segmentation_info = (
            segment_leaf(image_array)
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Original Image")
            st.image(
                processed_image,
                use_container_width=True
            )

        with col2:
            st.subheader("Binary Mask")
            st.image(
                leaf_mask,
                clamp=True,
                use_container_width=True
            )

        with col3:
            st.subheader("Segmented Leaf")
            st.image(
                segmented_image,
                use_container_width=True
            )

        st.subheader("Segmentation Statistics")

        st.metric(
            "Detected Leaf Area",
            f"{segmentation_info['leaf_area_percentage']}%"
        )

        st.write(
            "Leaf pixels:",
            segmentation_info["leaf_pixels"]
        )

        st.write(
            "Background pixels:",
            segmentation_info["background_pixels"]
        )
        
        
        st.divider()

        st.header("Leaf Feature Extraction")

        # Extract leaf features
        features = extract_features(
            image_array,
            leaf_mask
        )
        
        st.divider()

        st.header("Tomato Leaf Disease Classification")

        model_input = prepare_model_input(features)

        prediction = model.predict(model_input)[0]

        probabilities = model.predict_proba(model_input)[0]

        class_names = {
            0: "Healthy",
            1: "Early blight",
            2: "Late blight"
        }

        predicted_class = class_names[prediction]

        confidence = max(probabilities) * 100

        st.subheader("Prediction Result")

        if prediction == 0:
            st.success(f"Prediction: {predicted_class}")

        elif prediction == 1:
            st.warning(f"Prediction: {predicted_class}")

        else:
            st.error(f"Prediction: {predicted_class}")

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("Class-wise Prediction Probabilities")

        probability_data = pd.DataFrame({
            "Disease": [
                "Healthy",
                "Early blight",
                "Late blight"
            ],
            "Probability (%)": probabilities * 100
        })

        st.bar_chart(
            probability_data.set_index("Disease")
        )

        # Colour features
        st.subheader("1. Colour Features")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Dominant Colour",
                features["colour"]["dominant_colour"]
            )

            st.metric(
                "Mean Red",
                features["colour"]["mean_red"]
            )

        with col2:
            st.metric(
                "Mean Green",
                features["colour"]["mean_green"]
            )

            st.metric(
                "Mean Blue",
                features["colour"]["mean_blue"]
            )

        with col3:
            st.metric(
                "Mean Hue",
                features["colour"]["mean_hue"]
            )

            st.metric(
                "Mean Saturation",
                features["colour"]["mean_saturation"]
            )

        st.write("Colour distribution within detected mask:")

        st.json(features["colour"])

        # Shape features
        st.subheader("2. Shape Features")

        st.json(features["shape"])

        # Texture features
        st.subheader("3. Texture Features")

        st.json(features["texture"])

        st.subheader("4. Analysis Information")

        st.json(features["analysis_info"])
        
        
        st.divider()

        st.header("Leaf Descriptor Generation")

        # Generate structured descriptor
        descriptor = generate_descriptor(features)

        st.subheader("Colour Description")

        st.write(
            descriptor["colour"]["description"]
        )

        st.subheader("Shape Description")

        st.write(
            descriptor["shape"]["description"]
        )

        st.write(
            descriptor["shape"]["boundary_description"]
        )

        st.write(
            descriptor["shape"]["solidity_description"]
        )

        st.subheader("Texture Description")

        st.write(
            descriptor["texture"]["description"]
        )

        st.subheader("Overall Leaf Descriptor")

        st.success(
            descriptor["summary"]
        )

        st.warning(
            descriptor["disclaimer"]
        )

        st.subheader("Structured Descriptor Data")

        st.json(descriptor)

    except ValueError as error:

        st.error(str(error))

    except Exception as error:

        st.error(f"Unexpected error: {error}")


else:

    st.info("Please upload a leaf image to begin.")