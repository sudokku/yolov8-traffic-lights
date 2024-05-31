import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera

# Initialize the camera
camera = PiCamera()
camera.resolution = (224, 224)  # Set initial resolution
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(224, 224))

# Allow the camera to warm up
import time
time.sleep(0.1)

# Capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array

    # Resize the image to 224x224
    resized_image = cv2.resize(image, (224, 224))

    # Display the resized image
    cv2.imshow("Resized Frame", image)

    # Clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()
