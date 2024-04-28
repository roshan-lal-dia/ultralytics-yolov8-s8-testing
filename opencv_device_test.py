import cv2

# Check the first 5 indices.
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Device index {i} is available')
        
        # Capture a frame
        ret, frame = cap.read()
        if ret:
            # Save the frame as an image fileclear
            cv2.imwrite(f'device_{i}_frame.png', frame)
        
        cap.release()
    else:
        print(f'Device index {i} is not available')