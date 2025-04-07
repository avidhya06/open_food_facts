# write flask app to call all the classes and methods in the above code
import os
from flask import Flask, jsonify, request
app = Flask(__name__)  # Create an instance of the Flask application
from flask_cors import CORS
from text_data_preprocessor import DataCleaner
from image_data_preprocessor import ProductProcessor
from aws_image_data_downloader import ImageDataDownloader
from nutri_score_predictor import NutriScorePredictor
import json
from flask import send_file
import pandas as pd
import requests
import gzip
import logging
from datetime import datetime
from PIL import Image
# import from .env file
from dotenv import load_dotenv
load_dotenv()

data_keys_url = os.getenv("DATA_KEYS_URL")
input_text_file = os.getenv('INPUT_TEXT_FILE_PATH')
output_json_file = os.getenv('OUTPUT_JSON_PATH')
image_folder = os.getenv('IMAGE_FOLDER')
processed_image_folder = os.getenv('PROCESSED_IMAGE_FOLDER')
processed_text_file = os.getenv('PROCESSED_TEXT_FILE')

@app.route('/aws_image_data_downloader', methods=['POST'])
def aws_image_data_downloader():
    try:
        downloader = ImageDataDownloader(data_keys_url)
        downloader.run()
        return jsonify({"status": "Images downloaded successfully"})
    except Exception as e:
        logging.error(f"Error downloading images: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/text_data_processor', methods=['GET'])
def text_data_processor():
    try:
        cleaner = DataCleaner(
            input_file_path=os.getenv('INPUT_TEXT_FILE_PATH'),
            output_json_path=os.getenv('OUTPUT_JSON_PATH'),
            delimiter='\t',
            missing_threshold=0.7,
            chunk_size=50000,
            handle_bad_lines='skip',
            file_encoding='utf-8'
        )
        cleaner.process()
        return jsonify({"status": "Images downloaded successfully"})
    except Exception as e:
        logging.error(f"Text pre-process Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/image_data_processor', methods=['POST'])
def image_data_processor():
    try:

        if not os.path.isdir(image_folder):
            print(f"Error: Path provided is not a valid directory: {image_folder}")
        else:
            if not os.path.exists(processed_image_folder):
                os.makedirs(processed_image_folder)

            image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'))]

            for filename in image_files:
                image_path = os.path.join(image_folder, filename)
                processor = ProductProcessor(image_path)
                result = processor.process()

                output_filename = os.path.join(processed_image_folder, f"{os.path.splitext(filename)[0]}_extracted_data.json")
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)

        return jsonify({"status": "Data processed successfully"})
    
    except Exception as e:
        logging.error(f"Image pre-process Error: {e}")
        return jsonify({"error": str(e)}), 500
    
    
@app.route('/nutri_score_predictor', methods=['POST'])
def nutri_score_predictor():
    try:
        # Configuration
        config = {
            'JSONL_FILE_PATH': 'processed_text_file',
            'USE_INGREDIENTS_TEXT': True,
            'TEST_SET_SIZE': 0.25,
            'RANDOM_STATE': 42,
            'NUMERICAL_FEATURES': [
                'energy_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g',
                'sugars_100g', 'fiber_100g', 'proteins_100g', 'sodium_100g'
            ],
            'TEXT_FEATURE': 'ingredients_text',
            'TARGET_VARIABLE': 'nutrition_grade_fr',
            'VALID_TARGET_LABELS': ['a', 'b', 'c', 'd', 'e']
        }

        # Instantiate and run the predictor
        predictor = NutriScorePredictor(config)
        df = predictor.load_data()
        if df is not None:
            X_num, X_text, y = predictor.preprocess_data(df)
            if X_num is not None:
                X_num_train, X_num_test, X_text_train, X_text_test, y_train, y_test = predictor.split_data(X_num, X_text, y)
                X_train_final, X_test_final = predictor.transform_features(X_num_train, X_num_test, X_text_train, X_text_test)
                y_train_encoded, y_test_encoded = predictor.encode_target(y_train, y_test)
                predictor.train_model(X_train_final, y_train_encoded)
                predictor.evaluate_model(X_test_final, y_test_encoded)

        return jsonify({"status": "Model trained successfully"})
    except Exception as e:
        logging.error(f"Nutri Score Predictor Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    app.run(host='0.0.0.0', port=5000, debug=True)  # Run the Flask app on all available IPs and port 5000 in debug mode
    
