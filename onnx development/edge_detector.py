# edge_detector.py
import cv2
import json
import time
import threading
from kafka import KafkaProducer
from yolov8 import YOLOv8

class EdgeDetector:
    def __init__(self, model_path="mlmodel.onnx", kafka_broker="kafka-edge:9092"):
        # Initialize YOLOv8 model
        self.model = YOLOv8(model_path, conf_thres=0.6, iou_thres=0.4)
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_broker,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            compression_type='gzip'
        )
        
        # Metrics
        self.frame_count = 0
        self.start_time = time.time()

    def process_stream(self, camera_index=0):
        """Process video stream from camera."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError("Failed to open camera!")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame!")
                break
            
            # Perform detection
            detections = self._detect_objects(frame)
            
            # Send detections to Kafka
            self._send_to_kafka(detections)
            
            # Display results
            self._show_results(frame, detections)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def _detect_objects(self, frame):
        """Run YOLOv8 inference on a frame."""
        boxes, scores, class_ids = self.model(frame)
        return {
            "timestamp": time.time_ns(),
            "detections": [{
                "bbox": box.tolist(),
                "score": float(score),
                "class_id": int(class_id)
            } for box, score, class_id in zip(boxes, scores, class_ids)]
        }

    def _send_to_kafka(self, detections):
        """Send detection data to Kafka."""
        self.producer.send('lp-detections', value=detections)
        self.producer.flush()

    def _show_results(self, frame, detections):
        """Display detection results."""
        debug_frame = self.model.draw_detections(frame)
        cv2.imshow("Edge Detection", debug_frame)

    def start_metrics_collection(self, interval=5):
        """Collect and log system metrics."""
        import psutil
        while True:
            metrics = {
                "timestamp": time.time_ns(),
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "fps": self.frame_count / (time.time() - self.start_time)
            }
            self.producer.send('edge-metrics', value=metrics)
            time.sleep(interval)

if __name__ == "__main__":
    detector = EdgeDetector()
    
    # Start metrics collection in a separate thread
    metrics_thread = threading.Thread(target=detector.start_metrics_collection, daemon=True)
    metrics_thread.start()
    
    # Start processing video stream
    detector.process_stream()