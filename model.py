import cv2
import pytesseract
from matplotlib import pyplot as plt
import os
from openai import OpenAI
import json
from PIL import Image
client = OpenAI(
    # This is the default and can be omitted
    api_key= json.load(open("conf.json"))["api_key"]
)
def gpt(q=None):
  print("text")
  print(q)
  r = client.chat.completions.create(
      messages=[
          {
              "role": "system",
              "content": "You are an AI model that reads a text and returns an dict in JSON format. Extract all the ingredient details with its wight and percentage (two of them, if one is not exist , set null), without any extraneous words. For example, 'Sodium 120mg 5%' should be returned as {'Sodium': '120mg => 5%'}, and fix the caracters because the model maybe give wrong predecting of caracters.",
          } ,
          {
              "role": "user",
              "content": q,
          }
      ],
      model="gpt-4",
  )
  return r.choices[0].message.content

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to get a binary image
    _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Save the preprocessed image temporarily
    temp_image_path = "temp_image.png"
    cv2.imwrite(temp_image_path, binary_image)

    return temp_image_path
def preprocess_image2(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve thresholding
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply adaptive thresholding
    adaptive_threshold = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Save the preprocessed image temporarily
    temp_image_path = "temp_image.png"
    cv2.imwrite(temp_image_path, adaptive_threshold)

    return temp_image_path






    
# Load the image
def get_ingredient(image_path):
    custom_config = r'--tessdata-dir "/usr/share/tesseract-ocr/5/tessdata/"'

    # Load your image
    image = Image.open(preprocess_image(image_path))

    # Run OCR using the large English model
    text = pytesseract.image_to_string(image, lang='eng')
    return json.loads(gpt(text))
    # image = cv2.imread(image_path)

    # # Convert to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # Thresholding to preprocess the image
    # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # # Find contours to isolate the table
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Filter contours to find the table region
    # table_contour = None
    # for contour in contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     if w > 100 and h > 100:  # assuming the table is larger than 100x100 pixels
    #         table_contour = contour
    #         break

    # if table_contour is not None:
    #     x, y, w, h = cv2.boundingRect(table_contour)
    #     table_image = image[y:y+h, x:x+w]
    #     plt.imshow(cv2.cvtColor(table_image, cv2.COLOR_BGR2RGB))
    #     plt.show()

    #     # Use Tesseract to extract text from the isolated table image
    #     custom_config = r'--oem 3 --psm 6'
    #     table_text = pytesseract.image_to_string(table_image, config=custom_config)

        
    
