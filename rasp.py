import time
from picamera2 import Picamera2
import cv2

# Initialize the Picamera2 object
picam2 = Picamera2()

# Configure the camera for video capture
video_config = picam2.create_video_configuration()
picam2.configure(video_config)

# Start the camera
picam2.start()

# Create a window to display the video
cv2.namedWindow("Live Video Stream", cv2.WINDOW_AUTOSIZE)

try:
    while True:
        # Capture a frame from the camera
        frame = picam2.capture_array()

        # Display the frame in the window
        cv2.imshow("Live Video Stream", frame)

        # Check if the user pressed the 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Clean up and release resources
    cv2.destroyAllWindows()
    picam2.stop()
