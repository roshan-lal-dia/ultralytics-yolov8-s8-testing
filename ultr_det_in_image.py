import os
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Testing the trained model .pt using opencv


# Create a directory to save the cropped images
os.makedirs('cropped_number_plates', exist_ok=True)

# Load the pretrained model
model = YOLO('yolo8x-s8.pt')  # replace 'yolov5.pt' with the path to your YOLOv8 model file

# Load the image from file
image_path = 'detect_test.jpg'
frame = cv2.imread(image_path)

# Perform object detection on the image
results = model.predict(frame)

# Draw the detection results on the image
for r in results:
    annotator = Annotator(frame)
    boxes = r.boxes
    for i, box in enumerate(boxes):
        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
        c = box.cls
        class_name = model.names[int(c)]
        annotator.box_label(b, model.names[int(c)])
        
        # Check if the class name is 'book'
        if class_name == 'number_plate':
            # Crop the detected object from the image
            left, top, right, bottom = map(int, b)
            cropped = frame[top:bottom, left:right]
            
            # Save the cropped image to a file
            cv2.imwrite(f'cropped_number_plates/np_{i}.jpg', cropped)

# Display the image
cv2.imshow('Object Detection', frame)
cv2.waitKey(0)

# Destroy all windows
cv2.destroyAllWindows()
