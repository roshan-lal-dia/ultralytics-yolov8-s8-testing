from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

im = Image.open("cropped_number_plates/np_3.jpg")

text = pytesseract.image_to_string(im, lang = 'eng')

print(text)