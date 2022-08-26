import cv2
from time import sleep

import imutils
from matplotlib import pyplot as plt
import easyocr
import numpy as np
from PIL import Image
import pytesseract

#pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'

# Create our body classifier
plate_classifier = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Initiate video capture for video file
frame = cv2.imread('images/Cars9.png')


gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(bfilter, 30, 200)

plate = plate_classifier.detectMultiScale(gray, 1.1, 2)

keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

#print(contours)

#plt.imshow(cv2.cvtColor(contours, cv2.COLOR_BGR2RGB))


#plt.show()


# Extract bounding boxes for any bodies identified

try:
    cropped_image = frame[plate[0][1]:plate[0][1]+plate[0][3], plate[0][0]: plate[0][0] + plate[0][2]]

except:
    for (x, y, w, h) in plate:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

else:
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    text = result[0][-2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for (x, y, w, h) in plate:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, text=text, org=(x, y), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2,
                    lineType=cv2.LINE_AA)

# cropped_image = np.array(cropped_image)
# text = pytesseract.image_to_string(cropped_image)
# print(text)






plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.show()
