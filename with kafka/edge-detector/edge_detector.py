# edge-detector/edge_detector.py
import cv2
import json
import time
from kafka import KafkaProducer
from yolov8 import YOLOv8

# Initialize Kafka producer
producer = KafkaProducer(
    bootstrap_servers='kafka:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Initialize YOLOv8 object detector
model_path = "mlmodel.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

# Initialize the webcam
cap = cv2.VideoCapture(0)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():
    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Perform object detection
    boxes, scores, class_ids = yolov8_detector(frame)

    # Send detections to Kafka
    detection_data = {
        "timestamp": time.time_ns(),
        "detections": [{
            "bbox": box.tolist(),
            "score": float(score),
            "class_id": int(class_id)
        } for box, score, class_id in zip(boxes, scores, class_ids)]
    }
    producer.send('lp-detections', detection_data)

    # Display results
    combined_img = yolov8_detector.draw_detections(frame)
    cv2.imshow("Detected Objects", combined_img)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

producer.flush()
cap.release()
cv2.destroyAllWindows()