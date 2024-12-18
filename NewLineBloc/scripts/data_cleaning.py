import pandas as pd
import os
import re

# Define file paths
input_file = "/Users/mac/NewLineBloc/data/raw/contractor_original.csv"  
output_file = "/Users/mac/NewLineBloc/data/processed/cleaned_contractor_data.csv"

# Create directories if they don't exist
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv(input_file, delimiter=",", encoding="utf-8")

# Step 1: Remove duplicates
print("Removing duplicates...")
df = df.drop_duplicates()

# Step 2: Handle missing values and replace 'X' with 1, blanks with 0
# print("Handling missing values and transforming data...")

# # Replace all occurrences of 'X' (case-insensitive) with 1, and blanks (NaN) with 0
# def transform_missing_and_x(value):
#     if isinstance(value, str):
#         value = value.strip()  # Remove leading/trailing whitespace
#         if value.lower() == 'x':
#             return 1
#     return 0 if pd.isna(value) else value

# # Apply transformation to the entire DataFrame
# df = df.applymap(transform_missing_and_x)

# Step 3: Standardize text data
print("Standardizing text data...")
def clean_text(text):
    if isinstance(text, str):
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s\.\,]", "", text)  # Keep alphanumeric, space, dot, comma
        text = text.strip()  # Remove leading/trailing whitespaces
        return text
    return text

# Apply text cleaning to all object (string) columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].apply(clean_text)

# Step 4: Save cleaned data
print("Saving cleaned data...")
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"Data cleaning completed. Cleaned data saved to {output_file}")
