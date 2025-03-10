import cv2
import json
import time
from yolov8 import YOLOv8
from kafka import KafkaProducer

# Kafka Producer Configuration
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize YOLOv8 object detector
model_path = "mlmodel.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

detection_id = 1  # Unique ID for each detection

while cap.isOpened():
    # Read frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Update object localizer
    boxes, scores, class_ids = yolov8_detector(frame)

    # Prepare JSON output for all detected objects
    detections = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        detections.append({
            "id": detection_id,
            "box": {
                "x_min": int(box[0]),
                "y_min": int(box[1]),
                "x_max": int(box[2]),
                "y_max": int(box[3])
            },
            "score": float(score),
            "class_id": int(class_id),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        })
        detection_id += 1

    # If objects are detected, print and send JSON data
    if detections:
        json_output = {"detections": detections}
        print("\nDetected Objects:")
        print(json.dumps(json_output, indent=4))
        producer.send('input-topic', value=json_output)

    # Visualize detections on the frame
    combined_img = yolov8_detector.draw_detections(frame)
    cv2.imshow("Detected Objects", combined_img)

    # Press key 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
