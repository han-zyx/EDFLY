import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO  # Correct import for YOLOv8

# Load the YOLOv8 model
model = YOLO("mlmodel.pt")  # Adjust the path to your YOLOv8 model file

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 indicates the default camera

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Plot the results on the frame
    result_img = results[0].plot()

    # Display the frame with detections
    cv2.imshow("Object Detection", result_img)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
