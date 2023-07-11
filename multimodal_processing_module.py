import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import requests
from language_understanding_module import LanguageUnderstandingModule

class MultiModalProcessingModule:
    def __init__(self, api_key):
        self.image_processing = ImageProcessing()  # Initialize the ImageProcessing class for processing images
        self.text_processing = LanguageUnderstandingModule(api_key)
       
    def process_image_from_url(self, url):
        # Download the image from the provided URL
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))

        # Process the image using the ImageProcessing class
        processed_image = self.image_processing.process(img)

        return processed_image

    def process_text(self, text):
        # Process the text using the text_processing module
        processed_text = self.text_processing.process(text)

        return processed_text
        
class ImageProcessing:
    def __init__(self):
        pass

    def process(self, img):
        # Convert the image to grayscale
        grayscale_img = self.convert_to_grayscale(img)

        # Resize the image
        resized_img = self.resize_image(grayscale_img, new_width=128, new_height=128)

        return resized_img

    @staticmethod
    def convert_to_grayscale(img):
        # Convert the input image to grayscale using OpenCV
        img = np.asarray(img)
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return grayscale_img

    @staticmethod
    def resize_image(img, new_width, new_height):
        # Resize the input image to the specified dimensions using OpenCV
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_img