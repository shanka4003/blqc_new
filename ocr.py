import pytesseract
from PIL import Image
import cv2 as cv

# If Tesseract is not in PATH, you need to specify the full path
# Example for Windows: 
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load image from file

image = Image.open("2025-08-25_18-42-39_328.jpg")
image = image.rotate(90, expand=True)
# Extract text
text = pytesseract.image_to_string(image)

# cv.imshow("Image", image)

print("Extracted Text:")
print(text)
