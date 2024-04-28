#Testing the trained model .pt using opencv

import cv2
from ultralytics import YOLO

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
for i in range(len(results)):
    xywhcn = results[i]
    x, y, w, h, class_name, confidence = xywhcn
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()