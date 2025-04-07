import os
import pandas as pd
import requests
import gzip


class ImageDataDownloader:
    """Class to download and extract image data from AWS S3 bucket."""
    def __init__(self, data_keys_url, temp_folder="temp_images"):
        """
        Initialize the ImageDataDownloader with the URL for the data keys file
        and the temporary folder to store downloaded images.
        """
        self.data_keys_url = data_keys_url
        self.data_keys_file = "data_keys.gz"
        self.data_keys_file_extracted = "data_keys"
        self.temp_folder = temp_folder

    def download_data_keys(self):
        """
        Download the gzipped data keys file from the specified URL.
        If the file already exists, it will not download it again.
        """
        try:
            if not os.path.exists(self.data_keys_file):
                response = requests.get(self.data_keys_url)
                with open(self.data_keys_file, "wb") as file:
                    file.write(response.content)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading data keys: {e}")
            raise

    def extract_data_keys(self):
        """
        Extract the gzipped data keys file to a plain text file.
        If the extracted file already exists, it will not extract it again.
        """
        try:
            if not os.path.exists(self.data_keys_file_extracted):
                with gzip.open(self.data_keys_file, "rb") as file:
                    with open(self.data_keys_file_extracted, "wb") as extracted_file:
                        extracted_file.write(file.read())   
        except OSError as e:
            print(f"Error extracting data keys: {e}")
            raise

    def create_temp_folder(self):
        """
        Create a temporary folder to store downloaded images.
        If the folder already exists, it will not create it again.
        """
        if not os.path.exists(self.temp_folder):
            os.mkdir(self.temp_folder)

    def get_image_keys(self):
        """
        Read the extracted data keys file and filter for image keys
        that contain '.400.jpg'. Returns a DataFrame of filtered keys.
        """
        try:
            data_keys = pd.read_csv(self.data_keys_file_extracted, header=None, names=["key"])
            return data_keys[data_keys["key"].str.contains(".400.jpg")]
        except pd.errors.EmptyDataError:
            print("Error: The extracted data keys file is empty.")
            raise

    def download_images(self, image_keys):
        """
        Download images from the S3 bucket using the provided image keys.
        Images are saved in the temporary folder.
        """
        try:
            for index, row in image_keys.iterrows():
                image_key = row["key"]
                image_url = f"https://openfoodfacts-images.s3.eu-west-3.amazonaws.com/{image_key}"
                image_file = os.path.join(self.temp_folder, os.path.basename(image_key))
                if not os.path.exists(image_file):
                    response = requests.get(image_url, stream=True)
                    with open(image_file, "wb") as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            file.write(chunk)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading images: {e}")
            raise

    def run(self):
        """
        Main method to execute the entire image downloading process:
        1. Download the data keys file.
        2. Extract the data keys file.
        3. Create a temporary folder for images.
        4. Get the filtered image keys.
        5. Download the images using the keys.
        """
        self.download_data_keys()
        self.extract_data_keys()
        self.create_temp_folder()
        image_keys = self.get_image_keys()
        self.download_images(image_keys)
