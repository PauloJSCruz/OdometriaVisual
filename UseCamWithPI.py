from picamera2 import Picamera2
import cv2

picam2 = Picamera2()
picam2.start()
while True:
    image = picam2.capture_array()
    cv2.imshow('Custom Frame', image)
    cv2.waitKey(1)