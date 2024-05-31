import time
import io
from picamera import PiCamera
from picamera.array import PiRGBArray
from transformers import AutoModelForImageClassification, AutoProcessor
import torch
from PIL import Image
import numpy as np

# Load the pretrained model and processor from Hugging Face
model_name = 'google/vit-base-patch16-224'
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

# Function to preprocess the image for the model
def preprocess_image(image):
    # Convert the image to RGB (if it's not already in RGB format)
    image = image.convert('RGB')
    # Convert the image to a numpy array
    image_np = np.array(image)
    # Preprocess the image using the processor
    inputs = processor(images=image_np, return_tensors='pt')
    return inputs

# Function to perform inference on the image
def predict(image):
    inputs = preprocess_image(image)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

# Initialize the PiCamera
camera = PiCamera()
camera.resolution = (224, 224)
camera.framerate = 30

# Use PiRGBArray to capture frames from the camera
raw_capture = PiRGBArray(camera, size=(224, 224))

# Allow the camera to warm up
time.sleep(2)

try:
    # Capture frames continuously from the camera
    for frame in camera.capture_continuous(raw_capture, format='bgr', use_video_port=True):
        # Get the image from the frame
        image_np = frame.array
        
        # Convert the image to a PIL Image
        image = Image.fromarray(image_np)
        
        # Perform prediction on the captured image
        label = predict(image)
        
        # Print the prediction
        print(f'Predicted label: {label}')
        
        # Clear the stream for the next frame
        raw_capture.truncate(0)
        
        # Optionally, you can add a short sleep to control frame rate
        time.sleep(0.1)
        
except KeyboardInterrupt:
    # Gracefully handle the exit
    print("Exiting...")
finally:
    # Close the camera
    camera.close()
