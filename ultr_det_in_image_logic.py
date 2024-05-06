import csv
import time
import pytesseract
import os
import datetime
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from sklearn.metrics import jaccard_score
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Create a directory to save the cropped images
os.makedirs('cropped_number_plates', exist_ok=True)

# Load the pretrained model
model = YOLO('yolo8x-s8.pt')  # replace 'yolov5.pt' with the path to your YOLOv8 model file

# Load the image from file
image_path = 'detect_test.jpg'
frame = cv2.imread(image_path)

# Perform object detection on the image
results = model.predict(frame)

# Initialize prev_boxes as an empty list
prev_boxes = []

# Initialize a dictionary to store the last detection time for each rider
last_detection_time = {}

# Draw the detection results on the image
for r in results:
    annotator = Annotator(frame)
    boxes = r.boxes
    for i, box in enumerate(boxes):
        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
        c = box.cls
        class_name = model.names[int(c)]
        annotator.box_label(b, model.names[int(c)])

        # Check if the class name is 'rider'
        if class_name == 'rider':
            # Check if the rider is the same as in the previous frame
            if i < len(prev_boxes) and jaccard_score(b, prev_boxes[i]) > 0.5:
                # Check if the current timestamp is within the cooldown period
                if time.time() - last_detection_time.get(i, 0) < 3:
                    continue
                else:
                    # Update the last detection time for this rider
                    last_detection_time[i] = time.time()

            # Initialize a list to store violations
            violations = []

            # Check for violations
            for violation in ['cellphone', 'triple_riding']:
                violation_boxes = [box for box in boxes if model.names[int(box.cls)] == violation]
                if violation_boxes:
                    violations.append(violation)
                    for i, box in enumerate(violation_boxes):
                        # Crop the detected object from the image
                        left, top, right, bottom = map(int, box.xyxy[0])
                        cropped = frame[top:bottom, left:right]

                        # Save the cropped image to a file
                        img_path = f'cropped_violations/{violation}_{i}.jpg'
                        cv2.imwrite(img_path, cropped)

            # Check for 'without_helmet' violation
            helmet_violations = [box for box in boxes if model.names[int(box.cls)] == 'without_helmet']
            if helmet_violations:
                violations.append('without_helmet')
                for i, box in enumerate(helmet_violations):
                    # Crop the detected object from the image
                    left, top, right, bottom = map(int, box.xyxy[0])
                    cropped = frame[top:bottom, left:right]

                    # Save the cropped image to a file
                    img_path = f'cropped_violations/without_helmet_{i}.jpg'
                    cv2.imwrite(img_path, cropped)

            # If any violations are detected, crop the 'number_plate'
            if violations:
                for box in boxes:
                    if model.names[int(box.cls)] == 'number_plate':
                        # Crop the detected object from the image
                        left, top, right, bottom = map(int, box.xyxy[0])
                        cropped = frame[top:bottom, left:right]

                        # Save the cropped image to a file
                        img_path = f'cropped_number_plates/np_{i}.jpg'
                        cv2.imwrite(img_path, cropped)

                        # Preprocess the image for OCR

                        # Run OCR on the preprocessed image
                        config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                        license_plate_number = pytesseract.image_to_string(cropped, config=config)

                        # Convert the timestamp to a human-readable format
                        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

                        # Write the violation data to a CSV file
                        csv_file = 'violations.csv'
                        file_exists = os.path.isfile(csv_file)
                        with open(csv_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            if not file_exists:
                                writer.writerow(['Timestamp', 'Violations', 'Image Path', 'License Plate Number'])
                            writer.writerow([timestamp, violations, img_path, license_plate_number])

    # Update the previous boxes
    prev_boxes = [box.xyxy[0] for box in boxes]
# Display the image
cv2.imshow('Object Detection', frame)
cv2.waitKey(0)

# Destroy all windows
cv2.destroyAllWindows()