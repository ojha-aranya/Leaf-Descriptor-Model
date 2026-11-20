import pandas as pd
import numpy as np

df = pd.read_csv("dataset/processed/tomato_features.csv")

print("=" * 50)
print("FULL DATASET VALIDATION")
print("=" * 50)

print("\nDataset shape:", df.shape)

print("\nClass distribution:")
print(df["class_name"].value_counts())

print("\nMissing values:", df.isnull().sum().sum())

numeric_df = df.select_dtypes(include=np.number)
print("Infinite values:", np.isinf(numeric_df).sum().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print(
    "\nDuplicate filenames:",
    df["image_filename"].duplicated().sum()
)

print("\nUnique filenames:", df["image_filename"].nunique())

print("\nValidation completed.")