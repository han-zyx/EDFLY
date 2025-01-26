import cv2
from yolov8 import YOLOv8

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize YOLOv8 object detector
model_path = "mlmodel.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Update object localizer
    boxes, scores, class_ids = yolov8_detector(frame)

    combined_img = yolov8_detector.draw_detections(frame)
    cv2.imshow("Detected Objects", combined_img)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break









#use opencv------------------------------------------------------------------------------------


# import cv2
# from yolov8 import YOLOv8
# import pytesseract
# from collections import deque

# # Initialize the webcam
# cap = cv2.VideoCapture(0)

# # Initialize YOLOv8 object detector
# model_path = "mlmodel.onnx"
# yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

# # Set to store unique license plates
# unique_plates = set()

# # Queue to store recently detected plates
# recent_plates = deque(maxlen=10)  # Adjust maxlen based on frame rate

# cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

# while cap.isOpened():
#     # Read frame from the video
#     ret, frame = cap.read()

#     if not ret:
#         break

#     # Update object localizer
#     boxes, scores, class_ids = yolov8_detector(frame)

#     # Loop through detected objects
#     for box, score, class_id in zip(boxes, scores, class_ids):
#         if class_id == 2:  # Replace 2 with the correct class ID for license plates
#             x1, y1, x2, y2 = box.astype(int)

#             # Crop the license plate region
#             license_plate = frame[y1:y2, x1:x2]

#             # Preprocess the license plate image for OCR
#             gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
#             blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#             _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#             # Perform OCR to extract text
#             plate_text = pytesseract.image_to_string(binary, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

#             # Clean and validate the detected text
#             plate_text = ''.join(e for e in plate_text if e.isalnum())

#             # Add to the set if not already present
#             if plate_text and plate_text not in recent_plates:
#                 unique_plates.add(plate_text)
#                 recent_plates.append(plate_text)

#     # Draw detections on the frame
#     combined_img = yolov8_detector.draw_detections(frame)

#     # Display the count of unique license plates
#     cv2.putText(combined_img, f"Unique Plates: {len(unique_plates)}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     # Show the frame
#     cv2.imshow("Detected Objects", combined_img)

#     # Press key 'q' to stop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()








#flink integrated code------------------------------------------------------------------------------------
import cv2
from yolov8 import YOLOv8

# Initialize YOLOv8 object detector
model_path = "mlmodel.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible.")
else:
    print("Webcam is working.")
cap.release()

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a window to display the output
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Perform object detection on the frame
    boxes, scores, class_ids = yolov8_detector(frame)

    # Draw detections on the frame
    combined_img = yolov8_detector.draw_detections(frame)

    # Display the frame with detections
    cv2.imshow("Detected Objects", combined_img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()