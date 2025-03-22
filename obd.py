# import cv2
# import json
# import time
# from yolov8 import YOLOv8
# from kafka import KafkaProducer
# import pytesseract

# # Kafka Producer Configuration
# producer = KafkaProducer(
#     bootstrap_servers='localhost:9092',
#     value_serializer=lambda v: json.dumps(v).encode('utf-8')
# )

# # Initialize the video capture from file
# cap = cv2.VideoCapture("tc.mp4")

# # Initialize YOLOv8 object detector
# model_path = "mlmodel.onnx"
# yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

# cv2.namedWindow("Detected License Plates", cv2.WINDOW_NORMAL)

# # Define a single horizontal line for tracking re-identification avoidance
# line_y = 300  # Adjust based on camera position
# processed_plates = {}  # Store already processed license plates with timestamps
# cooldown_time = 5  # Time in seconds to prevent duplicate detections
# plate_counter = 1  # Unique ID counter for license plates

# def is_passing_line(y_min, y_max):
#     """Check if the object has passed the horizontal line."""
#     return y_min < line_y < y_max

# def extract_license_plate_text(plate_region):
#     """Extract text from the detected license plate using Tesseract OCR."""
#     gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
#     text = pytesseract.image_to_string(thresh, config='--psm 8')
#     return text.strip()

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Update object detector
#     boxes, scores, class_ids = yolov8_detector(frame)
    
#     # Prepare JSON output
#     detections = []
#     current_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
#     for box, score, class_id in zip(boxes, scores, class_ids):
#         x_min, y_min, x_max, y_max = map(int, box)
        
#         if is_passing_line(y_min, y_max):
#             plate_region = frame[y_min:y_max, x_min:x_max]
#             license_plate_text = extract_license_plate_text(plate_region)
            
#             # Avoid re-processing the same plate within cooldown time
#             if license_plate_text and (license_plate_text not in processed_plates or (time.time() - processed_plates[license_plate_text]) > cooldown_time):
#                 processed_plates[license_plate_text] = time.time()
#                 detections.append({
#                     "id": plate_counter,
#                     "time": current_time,
#                     "license_plate_text": license_plate_text
#                 })
#                 plate_counter += 1
    
#     # If license plates detected, send JSON data
#     if detections:
#         json_output = {"detections": detections}
#         print(json.dumps(json_output, indent=4))
#         producer.send('input-topic', value=json_output)
    
#     # Draw the detection line
#     cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
    
#     # Show the frame
#     cv2.imshow("Detected License Plates", frame)
    
#     # Press 'q' to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()





import cv2
import json
import time
from yolov8 import YOLOv8
from kafka import KafkaProducer
import pytesseract
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Kafka Producer Configuration
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Prometheus Metrics
start_http_server(8000)  # Expose metrics on port 8000
DETECTIONS_TOTAL = Counter('yolo_detections_total', 'Total number of license plates detected')
DETECTION_LATENCY = Histogram('yolo_detection_seconds', 'Time taken for YOLO detection', buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0])
FRAME_PROCESS_RATE = Gauge('yolo_frame_process_rate', 'Frames processed per second')
PLATES_PROCESSED = Counter('yolo_plates_processed_total', 'Total number of unique plates processed')

# Initialize the video capture from file
cap = cv2.VideoCapture("tc.mp4")
#cap = cv2.VideoCapture(0)

# Initialize YOLOv8 object detector
model_path = "mlmodel.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

cv2.namedWindow("Detected License Plates", cv2.WINDOW_NORMAL)

# Define a single horizontal line for tracking re-identification avoidance
line_y = 300  # Adjust based on camera position
processed_plates = {}  # Store already processed license plates with timestamps
cooldown_time = 5  # Time in seconds to prevent duplicate detections
plate_counter = 1  # Unique ID counter for license plates

def is_passing_line(y_min, y_max):
    """Check if the object has passed the horizontal line."""
    return y_min < line_y < y_max

def extract_license_plate_text(plate_region):
    """Extract text from the detected license plate using Tesseract OCR."""
    gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, config='--psm 8')
    return text.strip()

# Variables for frame rate calculation
frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame_start_time = time.time()

    # Update object detector with timing
    detection_start = time.time()
    boxes, scores, class_ids = yolov8_detector(frame)
    detection_end = time.time()
    DETECTION_LATENCY.observe(detection_end - detection_start)
    
    # Prepare JSON output
    detections = []
    current_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for box, score, class_id in zip(boxes, scores, class_ids):
        x_min, y_min, x_max, y_max = map(int, box)
        
        if is_passing_line(y_min, y_max):
            plate_region = frame[y_min:y_max, x_min:x_max]
            license_plate_text = extract_license_plate_text(plate_region)
            
            # Avoid re-processing the same plate within cooldown time
            if license_plate_text and (license_plate_text not in processed_plates or (time.time() - processed_plates[license_plate_text]) > cooldown_time):
                DETECTIONS_TOTAL.inc()  # Increment detection counter
                PLATES_PROCESSED.inc()  # Increment unique plates processed
                processed_plates[license_plate_text] = time.time()
                detections.append({
                    "id": plate_counter,
                    "time": current_time,
                    "license_plate_text": license_plate_text
                })
                plate_counter += 1
    
    # If license plates detected, send JSON data
    if detections:
        json_output = {"detections": detections}
        print(json.dumps(json_output, indent=4))
        producer.send('input-topic', value=json_output)
    
    # Calculate and update frame processing rate
    elapsed_time = time.time() - start_time
    if elapsed_time > 1.0:  # Update every second
        fps = frame_count / elapsed_time
        FRAME_PROCESS_RATE.set(fps)
        frame_count = 0
        start_time = time.time()

    # Draw the detection line
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
    
    # Show the frame
    cv2.imshow("Detected License Plates", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()