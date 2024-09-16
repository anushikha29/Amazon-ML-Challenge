import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from PIL import Image
import pytesseract
import cv2
import requests
from io import BytesIO

# Set the path for Tesseract (Linux path)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Define conversion factors and parsing functions
unit_conversion_factors = {
    'centimetre': 0.01, 'foot': 0.3048, 'inch': 0.0254, 'metre': 1, 'millimetre': 0.001, 'yard': 0.9144,
    'gram': 1e-3, 'kilogram': 1, 'microgram': 1e-9, 'milligram': 1e-6, 'ounce': 0.0283495, 'pound': 0.453592, 'ton': 1000,
    'kilovolt': 1000, 'millivolt': 1e-3, 'volt': 1,
    'kilowatt': 1000, 'watt': 1,
    'centilitre': 1e-2, 'cubic foot': 0.0283168, 'cubic inch': 1.63871e-5, 'cup': 0.24, 'decilitre': 1e-1,
    'fluid ounce': 0.0295735, 'gallon': 3.78541, 'imperial gallon': 4.54609, 'litre': 1, 'microlitre': 1e-6,
    'millilitre': 1e-3, 'pint': 0.473176, 'quart': 0.946353
}

def normalize_value(value, unit):
    if unit in unit_conversion_factors:
        return value * unit_conversion_factors[unit]
    return None

def parse_value(text):
    parts = text.split()
    if len(parts) == 2:
        try:
            value = float(parts[0])
            unit = parts[1]
            return value, unit
        except ValueError:
            return None, None
    return None, None

def preprocess_image(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply dilation and erosion to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    processed_img = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return processed_img

def download_image(url, save_path):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.save(save_path)

def extract_text_from_images(image_filenames):
    extracted_texts = []
    for filename in image_filenames:
        image_path = os.path.join('src/processed_images', filename)
        try:
            processed_img = preprocess_image(image_path)
            text = pytesseract.image_to_string(processed_img)
            extracted_texts.append(text.strip())
        except FileNotFoundError:
            print(f"File not found: {image_path}")
    return ' '.join(extracted_texts)  # Join text from multiple images

# Load training and test data
train_df = pd.read_csv(r'/home/nova/Documents/mlhackathon/student_resource 3/dataset/train.csv')
test_df = pd.read_csv(r'/home/nova/Documents/mlhackathon/student_resource 3/dataset/test.csv')

# Check if data is loaded correctly
if train_df.empty:
    raise ValueError("Training data is empty. Please check data preparation steps.")
if test_df.empty:
    raise ValueError("Test data is empty. Please check data preparation steps.")

# Print column names for debugging
print("Columns in test_df:")
print(test_df.columns)

# Adjust the column name to match your test data CSV
image_column = 'image_link'  # Correct column name based on the error message

# Create a directory to save downloaded images
download_dir = '/home/nova/Documents/mlhackathon/student_resource 3/src/downloaded_images'
os.makedirs(download_dir, exist_ok=True)

# Download images from URLs
for idx, row in test_df.iterrows():
    image_url = row[image_column]
    image_filename = os.path.join(download_dir, f'image_{idx}.jpg')
    download_image(image_url, image_filename)
    test_df.at[idx, 'local_image_path'] = image_filename

# Add the local image paths to the test dataframe
test_df['local_image_path'] = test_df.apply(lambda row: os.path.join(download_dir, f'image_{row.name}.jpg'), axis=1)

# Check for rows where extracted text will be None
print(f"Rows in test_df before text extraction: {len(test_df)}")

# Process training data
train_df['entity_name_encoded'] = train_df['entity_name'].astype('category').cat.codes
train_df['parsed_value'] = train_df['entity_value'].apply(lambda x: parse_value(x)[0])
train_df['unit'] = train_df['entity_value'].apply(lambda x: parse_value(x)[1])
train_df['normalized_value'] = train_df.apply(lambda row: normalize_value(row['parsed_value'], row['unit']), axis=1)

# Drop rows where 'normalized_value' is NaN
train_df = train_df.dropna(subset=['normalized_value'])

# Features and target variable
X_train = train_df[['entity_name_encoded']]
y_train = train_df['normalized_value']

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Process test data
test_df['entity_name_encoded'] = test_df['entity_name'].astype('category').cat.codes

# Extract text from images
test_df['extracted_text'] = test_df['local_image_path'].apply(lambda x: extract_text_from_images([x]))  # Single image per row

# Print debug info after text extraction
print(f"Extracted texts sample: {test_df[['extracted_text']].head()}")

# Process extracted text and normalize values
test_df['parsed_value'] = test_df['extracted_text'].apply(lambda x: parse_value(x)[0])
test_df['unit'] = test_df['extracted_text'].apply(lambda x: parse_value(x)[1])
test_df['normalized_value'] = test_df.apply(lambda row: normalize_value(row['parsed_value'], row['unit']), axis=1)

# Print debug info before dropping NaNs
print(f"Rows before dropping NaNs: {len(test_df)}")
print("Sample of normalized_value in test_df:")
print(test_df[['parsed_value', 'unit', 'normalized_value']].head())

# Drop rows where 'normalized_value' is NaN
test_df = test_df.dropna(subset=['normalized_value'])

# Print debug info after dropping NaNs
print(f"Rows after dropping NaNs: {len(test_df)}")

# Create X_test
X_test = test_df[['entity_name_encoded']]

# Check the shape and contents of X_test
print(f"X_test shape: {X_test.shape}")
print("X_test sample:")
print(X_test.head())

# Make predictions
if len(X_test) > 0:
    predictions = model.predict(X_test)
    test_df['predicted_value'] = predictions

    # Debug: Check number of predictions
    print(f"Number of predictions: {len(predictions)}")
    print(predictions[:10])
else:
    print("X_test is empty. No predictions can be made.")
    test_df['predicted_value'] = np.nan  # Ensure predicted_value column exists, even if empty

# Format predictions
test_df['prediction'] = test_df.apply(
    lambda row: f"{row['predicted_value']:.2f} {row['unit']}" if pd.notnull(row['predicted_value']) and row['unit'] else "",
    axis=1
)

# Prepare submission file
submission_df = test_df[['prediction']]
submission_df.index.name = 'index'  # Set the index name
submission_df.reset_index(inplace=True)  # Reset index to ensure it starts from 0

# Ensure that index matches the number of rows in the test data
submission_df = submission_df.iloc[:len(test_df)]  # Ensure it has the correct number of rows

# Save to CSV in a directory where you have write permissions
output_file = '/home/nova/Documents/mlhackathon/student_resource 3/test_predictions.csv'
submission_df.to_csv(output_file, index=False, encoding='utf-8')

print(f"Predictions saved to {output_file}")
