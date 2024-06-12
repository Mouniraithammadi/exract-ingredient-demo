import cv2
import pytesseract
from matplotlib import pyplot as plt
import os
from openai import OpenAI
import json
client = OpenAI(
    # This is the default and can be omitted
    api_key= json.load(open("conf.json"))["api_key"]
)
def gpt(q=None):
  r = client.chat.completions.create(
      messages=[
          {
              "role": "system",
              "content": "you a AI model that read a text and return a json text contains the ingredients that you extracted from the text and have its own % ,example : {'Sodium 120mg':'5%'}",
          } ,
          {
              "role": "user",
              "content": q,
          }
      ],
      model="gpt-3.5-turbo",
  )
  return r.choices[0].message.content
    
# Load the image
def get_ingredient(image_path):
    
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thresholding to preprocess the image
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours to isolate the table
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours to find the table region
    table_contour = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 100 and h > 100:  # assuming the table is larger than 100x100 pixels
            table_contour = contour
            break

    if table_contour is not None:
        x, y, w, h = cv2.boundingRect(table_contour)
        table_image = image[y:y+h, x:x+w]
        plt.imshow(cv2.cvtColor(table_image, cv2.COLOR_BGR2RGB))
        plt.show()

        # Use Tesseract to extract text from the isolated table image
        custom_config = r'--oem 3 --psm 6'
        table_text = pytesseract.image_to_string(table_image, config=custom_config)

        
    