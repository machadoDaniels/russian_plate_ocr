import cv2
from time import sleep
import imutils
from matplotlib import pyplot as plt
import easyocr
import numpy as np


def rescale_frame(frame_input, percent=75):
    width = int(frame_input.shape[1] * percent / 100)
    height = int(frame_input.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame_input, dim, interpolation=cv2.INTER_AREA)


# Create our body classifier
plate_classifier = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('License plate video.mp4')


# Loop once video is successfully loaded
while cap.isOpened():

    sleep(.05)
    # Read first frame
    ret, frame = cap.read()
    frame = rescale_frame(frame, 50)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    # Pass frame to our car classifier
    plate = plate_classifier.detectMultiScale(gray, 1.1, 3)
    if len(plate) != 0:

        # Extract bounding boxes for any bodies identified

        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]


        cropped_image = frame[plate[0][1]:plate[0][1] + plate[0][3], plate[0][0]: plate[0][0] + plate[0][2]]
        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)

        text = result[0][-2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        for (x, y, w, h) in plate:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, text=text, org=(x, y), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2,
                        lineType=cv2.LINE_AA)

    cv2.imshow('Cars', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
