
import pandas as pd

# Load extracted features
file_path = "dataset/processed/feature_validation_30.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("Dataset shape:", df.shape)
print("\nMissing values:", df.isnull().sum().sum())

print("\nClass distribution:")
print(df["class_name"].value_counts())


# Select important numerical features
selected_features = [
    "colour_green_percentage",
    "colour_yellow_percentage",
    "colour_brown_percentage",
    "colour_dark_percentage",
    "colour_mean_saturation",
    "shape_circularity",
    "shape_solidity",
    "texture_contrast",
    "texture_homogeneity",
    "texture_energy",
    "texture_correlation"
]

# Compare average feature values across classes
print("\n" + "=" * 60)
print("AVERAGE FEATURE VALUES BY CLASS")
print("=" * 60)

print(
    df.groupby("class_name")[selected_features]
    .mean()
    .round(3)
    .T
)

# Check feature variability
print("\n" + "=" * 60)
print("FEATURE STANDARD DEVIATION")
print("=" * 60)

print(
    df.groupby("class_name")[selected_features]
    .std()
    .round(3)
    .T
)

# Check for infinite values
print("\nInfinite values:",
      df.select_dtypes(include="number")
      .isin([float("inf"), float("-inf")])
      .sum().sum())

print("\nAnalysis completed.")
