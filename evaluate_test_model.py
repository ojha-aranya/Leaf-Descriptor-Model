
import pickle
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

processed_dir = Path("dataset/processed")

# Load test dataset
test = pd.read_csv(processed_dir / "test.csv")

# Load saved model
model_path = processed_dir / "tomato_random_forest.pkl"

with open(model_path, "rb") as file:
    model = pickle.load(file)

# Remove columns that were excluded during training
excluded_columns = [
    "image_filename",
    "class_name",
    "label",
    "analysis_info_analysed_pixels",
    "analysis_info_texture_method",
    "segmentation_leaf_pixels",
    "segmentation_background_pixels",
    "segmentation_leaf_area_percentage"
]

X_test = test.drop(columns=excluded_columns)
y_test = test["label"]

# Predict
predictions = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)

print("=" * 60)
print("FINAL TEST SET EVALUATION")
print("=" * 60)

print("Test images:", len(test))
print(f"Test accuracy: {accuracy:.4f}")
print(f"Test accuracy percentage: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=[
            "Healthy",
            "Early blight",
            "Late blight"
        ],
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1, 2]
    )
)

# Save test predictions
results = test[
    ["image_filename", "class_name", "label"]
].copy()

results["predicted_label"] = predictions

results.to_csv(
    processed_dir / "test_predictions.csv",
    index=False
)

print("\nTest predictions saved successfully.")
