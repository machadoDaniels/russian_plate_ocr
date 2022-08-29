import cv2
from time import sleep
from matplotlib import pyplot as plt
import easyocr
import numpy as np

# Create our body classifier
plate_classifier = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Initiate video capture for video file
frame = cv2.imread('images/Cars1.png')


gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
# edged = cv2.Canny(bfilter, 30, 200)

plate = plate_classifier.detectMultiScale(gray, 1.1, 3)

# keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours = imutils.grab_contours(keypoints)
# contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

#print(contours)

#plt.imshow(cv2.cvtColor(contours, cv2.COLOR_BGR2RGB))


#plt.show()


# Extract bounding boxes for any bodies identified



cropped_image = frame[plate[0][1]:plate[0][1]+plate[0][3], plate[0][0]: plate[0][0] + plate[0][2]]
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)
print(result)

text = result[0][-2]
font = cv2.FONT_HERSHEY_SIMPLEX

for (x, y, w, h) in plate:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, text=text, org=(x, y), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)


plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.show()
