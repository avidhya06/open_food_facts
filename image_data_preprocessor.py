import cv2
import pytesseract
from PIL import Image
import re
import json
from pyzbar import pyzbar
import requests
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import os


class ImageProcessor:
    def __init__(self, image_path):
        """Initializes the ImageProcessor with the given image path."""
        self.image_path = image_path
        self.color_image = None
        self.gray_image = None

    def preprocess_image(self):
        """Loads the image and converts it to grayscale for further processing."""
        try:
            self.color_image = cv2.imread(self.image_path)
            if self.color_image is None:
                print(f"Error: Image not found or could not be loaded: {self.image_path}")
                return False
            self.gray_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)
            return True
        except Exception as e:
            print(f"Error during preprocessing for {os.path.basename(self.image_path)}: {e}")
            return False


class OCRExtractor:
    @staticmethod
    def extract_text(gray_image):
        """Extracts text from the grayscale image using Tesseract OCR."""
        try:
            return pytesseract.image_to_string(gray_image, config='--psm 6')
        except pytesseract.TesseractNotFoundError:
            print("Error: Tesseract executable not found. Make sure it's installed and in your PATH.")
            raise SystemExit("Tesseract not found. Exiting.")
        except Exception as e:
            print(f"Error during OCR: {e}")
            return ""

    @staticmethod
    def post_process_text(text):
        """Cleans up the raw OCR text by removing empty lines and extra spaces."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return "\n".join(lines)


class BarcodeExtractor:
    @staticmethod
    def extract_barcodes(gray_image):
        """Detects and decodes barcodes from the grayscale image using pyzbar."""
        barcodes_found = []
        try:
            barcodes = pyzbar.decode(gray_image)
            for barcode in barcodes:
                barcode_data = barcode.data.decode('utf-8')
                barcode_type = barcode.type
                barcodes_found.append({"data": barcode_data, "type": barcode_type})
            return barcodes_found
        except Exception as e:
            print(f"Error during barcode detection: {e}")
            return []


class TextEntityExtractor:
    @staticmethod
    def extract_entities(text):
        """Extracts structured entities such as size and ingredients from the OCR text."""
        entities = {}
        if not text:
            return entities

        # Extract potential weights/volumes
        matches = re.findall(r'\b(\d+(?:[.,]\d+)?)\s*(ml|l|g|kg|oz|fl\.?\s*oz\.?)\b', text, re.IGNORECASE)
        if matches:
            unit = matches[0][1].lower()
            unit = {'ml': 'ml', 'l': 'l', 'g': 'g', 'kg': 'kg', 'oz': 'oz', 'fl oz': 'fl oz'}.get(unit, unit)
            entities['size'] = f"{matches[0][0].replace(',', '.')} {unit}"

        # Extract ingredients
        ingredients_match = re.search(r'(ingredients?|composition)\s*:', text, re.IGNORECASE)
        if ingredients_match:
            entities['has_ingredients_list'] = True
            start_index = ingredients_match.end()
            potential_ingredients_text = text[start_index:].split('\n\n')[0]
            potential_ingredients_lines = [line.strip() for line in potential_ingredients_text.split('\n') if line.strip()]
            entities['potential_ingredients_preview'] = potential_ingredients_lines[:5]

        return entities


class VisualEntityExtractor:
    @staticmethod
    def get_dominant_colors(color_image, k=3):
        """Identifies the k most dominant colors in the image using KMeans clustering."""
        try:
            h, w, _ = color_image.shape
            scale = 1.0
            max_dim = 300
            if h > max_dim or w > max_dim:
                scale = max_dim / max(h, w)
                small_img = cv2.resize(color_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            else:
                small_img = color_image

            pixels = small_img.reshape((-1, 3))
            pixels = np.float32(pixels)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)

            label_counts = Counter(labels.flatten())
            dominant_colors_bgr = [centers[i] for i, _ in label_counts.most_common(k)]

            dominant_colors_hex = []
            for color in dominant_colors_bgr:
                b, g, r = [int(max(0, min(255, c))) for c in color]
                hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
                dominant_colors_hex.append(hex_color)

            return dominant_colors_hex
        except Exception as e:
            print(f"Error extracting dominant colors: {e}")
            return []


class OpenFoodFactsLookup:
    @staticmethod
    def lookup(barcode_data):
        """Fetches product information from Open Food Facts using the barcode data."""
        if not barcode_data:
            return None
        try:
            url = f"https://world.openfoodfacts.net/api/v2/product/{barcode_data}?fields=product_name,brands,categories_tags,ingredients_text_en,nutriments,image_url"
            headers = {'User-Agent': 'ProductExtractorScript/1.0 (Language=Python)'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == 1 and "product" in data:
                product_info = data.get("product", {})
                categories = [cat.split(':')[-1].replace('-', ' ') for cat in product_info.get("categories_tags", [])]
                return {
                    "off_product_name": product_info.get("product_name"),
                    "off_brands": product_info.get("brands"),
                    "off_categories": categories,
                    "off_ingredients_text": product_info.get("ingredients_text_en"),
                    "off_nutriments": product_info.get("nutriments"),
                    "off_image_url": product_info.get("image_url"),
                    "off_found": True
                }
            else:
                return {"off_found": False, "barcode_searched": barcode_data}
        except Exception as e:
            print(f"Error during Open Food Facts API request: {e}")
            return None


class ProductProcessor:
    def __init__(self, image_path):
        """Initializes the ProductProcessor with the given image path."""
        self.image_path = image_path
        self.processor = ImageProcessor(image_path)
        self.ocr_text = ""
        self.cleaned_text = ""
        self.barcodes = []
        self.text_entities = {}
        self.visual_entities = {}
        self.off_data = {"off_found": False}

    def process(self):
        """Processes the image to extract OCR text, barcodes, entities, and visual features."""
        if not self.processor.preprocess_image():
            return {"source_image": os.path.basename(self.image_path), "processing_status": "Error: Image loading/preprocessing failed"}

        try:
            self.ocr_text = OCRExtractor.extract_text(self.processor.gray_image)
            self.cleaned_text = OCRExtractor.post_process_text(self.ocr_text)
            self.barcodes = BarcodeExtractor.extract_barcodes(self.processor.gray_image)
            self.text_entities = TextEntityExtractor.extract_entities(self.cleaned_text)
            self.visual_entities = {"dominant_colors_hex": VisualEntityExtractor.get_dominant_colors(self.processor.color_image)}
            if self.barcodes:
                self.off_data = OpenFoodFactsLookup.lookup(self.barcodes[0]['data'])
        except Exception as e:
            print(f"Error during processing: {e}")
            return {"source_image": os.path.basename(self.image_path), "processing_status": f"Error: {e}"}

        return self.consolidate_information()

    def consolidate_information(self):
        """Consolidates all extracted information into a single dictionary."""
        final_output = {
            "source_image": os.path.basename(self.image_path),
            "processing_status": "Success",
            "ocr_raw_text": self.ocr_text,
            "extracted_entities": self.text_entities,
            "detected_barcodes": self.barcodes,
            "visual_features": self.visual_entities,
            "openfoodfacts_data": self.off_data
        }
        return final_output
