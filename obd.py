import cv2
import pandas as pd
import numpy as np
from datetime import datetime
from yolov8 import YOLOv8
import easyocr
import re
import json
import time
from kafka import KafkaProducer
from prometheus_client import Counter as PrometheusCounter, Histogram, Gauge, start_http_server
import imutils

#kafka producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Prometheus metrics
start_http_server(8000)
DETECTIONS_TOTAL = PrometheusCounter('yolo_detections_total', 'Total number of license plates detected')
DETECTION_LATENCY = Histogram('yolo_detection_seconds', 'Time taken for YOLO detection')
PLATES_PROCESSED = PrometheusCounter('yolo_plates_processed_total', 'Total number of unique plates processed')

cap = cv2.VideoCapture('sample.mp4')

# Load YOLOv8 model
model_path = "mlmodel.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

reader = easyocr.Reader(['en'], verbose=False)

area = [(1, 440), (1, 400), (1019, 400), (1019, 440)]

cv2.namedWindow('RGB')

processed_numbers = set()
plate_counter = 1

with open("car_plate_data.txt", "a") as file:
    file.write("NumberPlate\tDateTime\n")

def improved_crop_plate(plate_region):
    """plate cropping"""
    try:
        gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)
 
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break

        if location is not None:
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [location], 0, 255, -1)
            (x, y) = np.where(mask == 255)
            if len(x) > 0 and len(y) > 0:
                (x1_c, y1_c) = (np.min(x), np.min(y))
                (x2_c, y2_c) = (np.max(x), np.max(y))
                cropped_image = gray[x1_c:x2_c+3, y1_c:y2_c+3]
                return cropped_image
        return plate_region
    except Exception:
        return plate_region

def extract_license_plate_text(plate_region):
    """Extract text from plates"""
    try:
        processed_image = improved_crop_plate(plate_region)
       
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY) if len(processed_image.shape) == 3 else processed_image
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
       
        ocr_result = reader.readtext(bfilter)
        if ocr_result:
            text = ocr_result[0][1].strip()
            text = text.replace('(', '').replace(')', '').replace(',', '').replace(']','')
            return text
        return ""
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return ""

def format_plate_text(text):
    """Format the plate"""
    text = re.sub(r'[^A-Z0-9]', '', text).upper()
    
    patterns = [
        r'^([A-Z]{3})(\d{4})$',  
        r'^([A-Z]{2})(\d{4})$', 
        r'^([1-9]{2})(\d{4})$'  
    ]
    
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            letters, numbers = match.groups()
            return f"{letters} {numbers}"
   
    match = re.search(r'([A-Z]{1,3})(\d{4})', text)
    return f"{match.group(1)} {match.group(2)}" if match else ""

while True:    
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (1020, 500))
   
    detection_start = time.time()
    boxes, scores, class_ids = yolov8_detector(frame)
    detection_end = time.time()
    DETECTION_LATENCY.observe(detection_end - detection_start)
    
    detections = []
    
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
       
        x1 = max(0, x1 - 1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1]-1, x2)
        y2 = min(frame.shape[0]-1, y2)
       
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
      
        if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
              
                if crop.size == 0:
                    continue
                
                text = extract_license_plate_text(crop)
                
                if text:
                    formatted_text = format_plate_text(text)
                    if formatted_text and formatted_text not in processed_numbers:
                        DETECTIONS_TOTAL.inc()
                        PLATES_PROCESSED.inc()
                        processed_numbers.add(formatted_text)
                        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        with open("car_plate_data.txt", "a") as file:
                            file.write(f"{formatted_text}\t{current_datetime}\n")
                      
                        detections.append({
                            "id": plate_counter,
                            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time())),
                            "license_plate_text": formatted_text
                        })
                        plate_counter += 1
                       
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, formatted_text, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow('crop', crop)
    
    if detections:
        json_output = {"detections": detections}
        print("Detected Objects:")
        print(json.dumps(json_output, indent=4))
        producer.send('input-topic', value=json_output)
   
    combined_img = yolov8_detector.draw_detections(frame)
    cv2.polylines(combined_img, [np.array(area, np.int32)], True, (255, 0, 0), 2)
    cv2.imshow("RGB", combined_img)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()    
cv2.destroyAllWindows()


