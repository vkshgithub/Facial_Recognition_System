import cv2
import pytesseract
import easyocr
import numpy as np
from matplotlib import pyplot as plt

image_path = 'sign.jpg'

reader = easyocr.Reader(['en'], gpu=True)
result = reader.readtext(image_path)
# print(result)

img = cv2.imread(image_path)
font = cv2.FONT_HERSHEY_SIMPLEX
spacer = 100
for detection in result:
    top_left = tuple(detection[0][0])
    bottom_right = tuple(detection[0][2])
    text = detection[1]
    img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)
    img = cv2.putText(img, text, (20, spacer), font,
                      0.5, (0, 255, 0), 2, cv2.LINE_AA)
    spacer += 15

plt.imshow(img)
plt.show()
