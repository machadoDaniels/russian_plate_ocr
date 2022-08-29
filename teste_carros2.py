import cv2
from time import sleep


def rescale_frame(frame_input, percent=75):
    width = int(frame_input.shape[1] * percent / 100)
    height = int(frame_input.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame_input, dim, interpolation=cv2.INTER_AREA)


# Create our body classifier
plate_classifier = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('License plate video.mp4')
cont = 0
# Loop once video is successfully loaded
while cap.isOpened():

    sleep(.05)
    # Read first frame
    ret, frame = cap.read()
    frame = rescale_frame(frame, 50)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

    if cont == 1:
        plt.imshow(bfilter)
    # Pass frame to our car classifier
    plate = plate_classifier.detectMultiScale(gray, 1.1, 3)
    if len(plate) == 1:


    # Extract bounding boxes for any bodies identified

    for (x, y, w, h) in plate:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('Cars', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
