import os
import sys
import pandas as pd
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
import re
from dotenv import load_dotenv
import cv2
import numpy as np

# Load environment variables
load_dotenv()
API_KEY = os.getenv('GOOGLE_CLOUD_API_KEY')

# Initialize Google Cloud Vision client
client = vision_v1.ImageAnnotatorClient()

class EntityExtractor:
    def __init__(self):
        self.patterns = {
                'width': r'(\d+(?:\.\d+)?)\s*(cm|mm|m|inch|foot)',
                'depth': r'(\d+(?:\.\d+)?)\s*(cm|mm|m|inch|foot)',
                'height': r'(\d+(?:\.\d+)?)\s*(cm|mm|m|inch|foot)',
                'maximum_weight_recommendation': r'(\d+(?:\.\d+)?)\s*(kg|g|lb|pound|ton)',
                'voltage': r'(\d+(?:\.\d+)?)\s*(v|volt|kilovolt)',
                'wattage': r'(\d+(?:\.\d+)?)\s*(w|watt|kilowatt)',
                'item_weight': r'(\d+(?:\.\d+)?)\s*(kg|g|lb|pound|ton)',
        }

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        gray = cv2.dilate(gray, kernel, iterations=1)
        return gray

    def extract_text(self, image):
        # Convert the preprocessed image to byte content
        _, buffer = cv2.imencode('.jpg', image)
        content = buffer.tobytes()

        image = types.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations
        if texts:
            return texts[0].description
        return ""

    def extract_values(self, text):
        values = {key: [] for key in self.patterns.keys()}
        for entity_name, pattern in self.patterns.items():
            matches = re.findall(pattern, text.lower())
            if matches:
                values[entity_name] = [f"{float(value):.2f} {unit}" for value, unit in matches]
        return values

    def match_value(self, entity_name, values):
        if entity_name in values and values[entity_name]:
            # Implement logic to select the most appropriate match based on entity type
            return values[entity_name][0]  # Or apply additional logic if needed
        return ""

    def process_image(self, image_path, entity_name):
        preprocessed_image = self.preprocess_image(image_path)
        text = self.extract_text(preprocessed_image)
        values = self.extract_values(text)
        return self.match_value(entity_name, values)

def main():
    # Load test data
    test_data = pd.read_csv('../dataset/sample_test.csv')
    
    # Set the path to the existing processed images
    image_dir = '../src/processed_images'
    
    # Initialize the EntityExtractor
    extractor = EntityExtractor()
    
    results = []
    
    for _, row in test_data.iterrows():
        index = row['index']
        image_filename = os.path.basename(row['image_link'])
        entity_name = row['entity_name']
        image_path = os.path.join(image_dir, image_filename)
        
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            results.append({'index': index, 'prediction': ""})
            continue
        
        try:
            prediction = extractor.process_image(image_path, entity_name)
            results.append({'index': index, 'prediction': prediction})
            
            print(f"Processed image {index}: Entity: {entity_name}, Prediction: {prediction}")
            
            if len(results) % 10 == 0:
                print(f"Processed {len(results)} images")
        except Exception as e:
            print(f"Error processing image for index {index}: {str(e)}")
            results.append({'index': index, 'prediction': ""})
    
    # Create output DataFrame
    output_df = pd.DataFrame(results)
    
    # Ensure all indices are present
    all_indices = set(test_data['index'])
    missing_indices = all_indices - set(output_df['index'])
    for idx in missing_indices:
        output_df = output_df.append({'index': idx, 'prediction': ""}, ignore_index=True)
    
    # Sort by index
    output_df = output_df.sort_values('index').reset_index(drop=True)
    
    # Save to CSV
    output_file = 'test_out2.csv'
    output_df.to_csv(output_file, index=False)
    
    print(f"Results saved to {output_file}")
    print(f"Output file contains {len(output_df)} rows")

if __name__ == "__main__":
    main()
