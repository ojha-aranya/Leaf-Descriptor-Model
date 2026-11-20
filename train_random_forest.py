import pickle
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. Load datasets
# --------------------------------------------------

processed_dir = Path("dataset/processed")

train = pd.read_csv(processed_dir / "train.csv")
validation = pd.read_csv(processed_dir / "validation.csv")

print("=" * 60)
print("RANDOM FOREST BASELINE CLASSIFIER")
print("=" * 60)

print("Training images:", len(train))
print("Validation images:", len(validation))

# --------------------------------------------------
# 2. Select input features
# --------------------------------------------------

# Features we do not want the model to learn from
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

X_train = train.drop(columns=excluded_columns)
y_train = train["label"]

X_val = validation.drop(columns=excluded_columns)
y_val = validation["label"]

# --------------------------------------------------
# 3. Identify numerical and categorical features
# --------------------------------------------------

categorical_features = ["colour_dominant_colour"]

numerical_features = [
    column for column in X_train.columns
    if column not in categorical_features
]

print("\nNumerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))
print("Total input features:", X_train.shape[1])

# --------------------------------------------------
# 4. Preprocess categorical features
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# --------------------------------------------------
# 5. Create Random Forest model
# --------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# --------------------------------------------------
# 6. Create complete pipeline
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", random_forest)
    ]
)

# --------------------------------------------------
# 7. Train model
# --------------------------------------------------

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Training completed!")

# --------------------------------------------------
# Save trained model
# --------------------------------------------------

model_path = processed_dir / "tomato_random_forest.pkl"

with open(model_path, "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully:", model_path)

# --------------------------------------------------
# 8. Evaluate on validation set
# --------------------------------------------------

predictions = model.predict(X_val)

# --------------------------------------------------
# Save validation predictions for error analysis
# --------------------------------------------------

validation_results = validation[
    ["image_filename", "class_name", "label"]
].copy()

validation_results["predicted_label"] = predictions

label_names = {
    0: "Healthy",
    1: "Early blight",
    2: "Late blight"
}

validation_results["predicted_class"] = (
    validation_results["predicted_label"].map(label_names)
)

validation_results["correct"] = (
    validation_results["label"]
    == validation_results["predicted_label"]
)

# Save all predictions
validation_results.to_csv(
    processed_dir / "validation_predictions.csv",
    index=False
)

# Save only incorrect predictions
misclassified = validation_results[
    validation_results["correct"] == False
]

misclassified.to_csv(
    processed_dir / "misclassified_validation.csv",
    index=False
)

print("\nIncorrect predictions:", len(misclassified))
print(
    "Saved misclassified images:",
    processed_dir / "misclassified_validation.csv"
)

accuracy = accuracy_score(y_val, predictions)

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_val,
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
        y_val,
        predictions,
        labels=[0, 1, 2]
    )
)

print("\nBaseline model evaluation completed.")


# --------------------------------------------------
# 9. Feature importance analysis
# --------------------------------------------------

importances = model.named_steps["classifier"].feature_importances_

feature_names = (
    model.named_steps["preprocessor"].get_feature_names_out()
)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\n" + "=" * 60)
print("TOP 15 IMPORTANT FEATURES")
print("=" * 60)

print(importance_df.head(15).to_string(index=False))

importance_df.to_csv(
    processed_dir / "feature_importance.csv",
    index=False
)

print("\nFeature importance saved successfully.")
