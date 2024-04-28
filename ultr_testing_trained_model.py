#Testing the trained model .pt using opencv

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Load the pretrained model
model = YOLO('yolov8n.pt')  # replace 'yolov5.pt' with the path to your YOLOv8 model file

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Perform object detection on the frame
    results = model.predict(frame)

    # Draw the detection results on the frame
    for r in results:
        
        annotator = Annotator(frame)
        
        boxes = r.boxes
        for box in boxes:
            
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()