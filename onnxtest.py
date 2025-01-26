import cv2
from ultralytics import YOLO

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the YOLOv8 model
model_path = "mlmodel.onnx"
yolov8_detector = YOLO(model_path)  # Load the ONNX model

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():
    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Perform inference with confidence and IoU thresholds
    results = yolov8_detector(frame, conf=0.5, iou=0.5)  # Set conf and iou thresholds here

    # Visualize the results
    annotated_frame = results[0].plot()  # Draw detections on the frame

    # Display the annotated frame
    cv2.imshow("Detected Objects", annotated_frame)

    # Press key 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()