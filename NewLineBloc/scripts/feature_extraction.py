import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import os
import re

# Define file paths
input_path = "/Users/mac/NewLineBloc/data/processed/cleaned_contractor_data.csv"
output_path = "/Users/mac/NewLineBloc/data/processed/extracted_features.csv"

# Load the dataset
df = pd.read_csv(input_path)

# Step 1: Drop irrelevant or redundant columns
irrelevant_columns = ["Zip", "Website", "Emails", "Phones", "Street", "Indian Economic Enterprise", "Manufacturer of Goods"]
df = df.drop(columns=irrelevant_columns, errors='ignore')

# Step 2: Handle missing values
print("Handling missing values...")
# Replace 'X' with 1 and blanks with 0 in binary ownership columns
binary_columns = [
    "Foreign Owned", "Minority-Owned Business", "Veteran-Owned Business",
    "Women-Owned Small Business", "Economically Disadvantaged Women-Owned Small Business (EDWOSB) Joint Venture"
]

def binary_transform(value):
    if isinstance(value, str):
        value = value.strip().lower()
        return 1 if value == 'x' else 0
    return 0

df[binary_columns] = df[binary_columns].applymap(binary_transform)

# Fill remaining NaN values with 0
df = df.fillna(0)

# Step 3: Vectorize textual columns
text_columns = ["Company Name", "Industry Title", "Industry"]
def vectorize_column(column_name, df, vectorizer=None):
    print(f"Vectorizing column: {column_name}")
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')

    # Replace missing values with an empty string
    df[column_name] = df[column_name].fillna("")

    # Fit and transform the column
    tfidf_matrix = vectorizer.fit_transform(df[column_name])

    # Create a DataFrame from the TF-IDF matrix
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"{column_name}_{feature}" for feature in vectorizer.get_feature_names_out()]
    )

    return tfidf_df

# Apply TF-IDF vectorization
vectorized_features = []
for col in text_columns:
    tfidf_df = vectorize_column(col, df)
    vectorized_features.append(tfidf_df)

# Step 4: Encode categorical columns
categorical_columns = ["State", "City", "NAICS Code"]
def encode_categorical_columns(columns, df):
    print("Encoding categorical columns:", columns)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Replace missing values with "Unknown"
    df[columns] = df[columns].fillna("Unknown")

    # Fit and transform the columns
    encoded_array = encoder.fit_transform(df[columns])

    # Create a DataFrame from the encoded array
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(columns)
    )

    return pd.concat([df.drop(columns, axis=1), encoded_df], axis=1)

categorical_encoded_df = encode_categorical_columns(categorical_columns, df)

# Step 5: Handle Business Designations
print("Handling Business Designations...")
if "Business Designations" in df.columns:
    designation_column = "Business Designations"
    df[designation_column] = df[designation_column].fillna("")
    designations = df[designation_column].str.split('|').explode().unique()
    for designation in designations:
        if pd.notna(designation):
            df[f"designation_{designation.strip()}"] = df[designation_column].apply(lambda x: 1 if designation in str(x).split('|') else 0)
    df = df.drop(columns=[designation_column])

# Combine vectorized features with the main DataFrame
print("Combining features...")
final_features = pd.concat([categorical_encoded_df] + vectorized_features, axis=1)

# Step 6: Save the processed dataset
os.makedirs(os.path.dirname(output_path), exist_ok=True)
final_features.to_csv(output_path, index=False)
print(f"Feature extraction completed. Saved to {output_path}")
