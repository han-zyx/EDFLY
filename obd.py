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





# import cv2
# import json
# import time
# from yolov8 import YOLOv8
# from kafka import KafkaProducer
# import pytesseract
# from prometheus_client import Counter, Histogram, Gauge, start_http_server

# # Kafka Producer Configuration
# producer = KafkaProducer(
#     bootstrap_servers='localhost:9092',
#     value_serializer=lambda v: json.dumps(v).encode('utf-8')
# )

# # Prometheus Metrics
# start_http_server(8000)  # Expose metrics on port 8000
# DETECTIONS_TOTAL = Counter('yolo_detections_total', 'Total number of license plates detected')
# DETECTION_LATENCY = Histogram('yolo_detection_seconds', 'Time taken for YOLO detection', buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0])
# FRAME_PROCESS_RATE = Gauge('yolo_frame_process_rate', 'Frames processed per second')
# PLATES_PROCESSED = Counter('yolo_plates_processed_total', 'Total number of unique plates processed')

# # Initialize the video capture from file
# cap = cv2.VideoCapture("sample.mp4")
# #cap = cv2.VideoCapture(0)

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

# # Variables for frame rate calculation
# frame_count = 0
# start_time = time.time()

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_count += 1
#     frame_start_time = time.time()

#     # Update object detector with timing
#     detection_start = time.time()
#     boxes, scores, class_ids = yolov8_detector(frame)
#     detection_end = time.time()
#     DETECTION_LATENCY.observe(detection_end - detection_start)
    
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
#                 DETECTIONS_TOTAL.inc()  # Increment detection counter
#                 PLATES_PROCESSED.inc()  # Increment unique plates processed
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
    
#     # Calculate and update frame processing rate
#     elapsed_time = time.time() - start_time
#     if elapsed_time > 1.0:  # Update every second
#         fps = frame_count / elapsed_time
#         FRAME_PROCESS_RATE.set(fps)
#         frame_count = 0
#         start_time = time.time()

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




# import cv2
# import json
# import time
# from yolov8 import YOLOv8
# from kafka import KafkaProducer
# import pytesseract
# from prometheus_client import Counter as PrometheusCounter, Histogram, Gauge, start_http_server
# from collections import Counter

# # Kafka Producer Configuration
# producer = KafkaProducer(
#     bootstrap_servers='localhost:9092',
#     value_serializer=lambda v: json.dumps(v).encode('utf-8')
# )

# # Prometheus Metrics
# start_http_server(8000)
# DETECTIONS_TOTAL = PrometheusCounter('yolo_detections_total', 'Total number of license plates detected')
# DETECTION_LATENCY = Histogram('yolo_detection_seconds', 'Time taken for YOLO detection', buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0])
# FRAME_PROCESS_RATE = Gauge('yolo_frame_process_rate', 'Frames processed per second')
# PLATES_PROCESSED = PrometheusCounter('yolo_plates_processed_total', 'Total number of unique plates processed')

# # Initialize video capture
# cap = cv2.VideoCapture("tc.mp4")
# frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

# # Initialize YOLOv8 detector
# model_path = "mlmodel2.onnx"
# yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

# cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

# # Detection tracking
# plate_counter = 1  # Unique ID for each plate
# active_plates = {}  # Dictionary to track active plates: {plate_id: {'texts': [], 'last_seen': time, 'position': (x, y), 'size': (w, h)}}

# def extract_license_plate_text(plate_region):
#     """Extract text from detected license plate using Tesseract OCR with preprocessing."""
#     gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
#     # Apply preprocessing to improve OCR accuracy
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
#     _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
#     text = pytesseract.image_to_string(thresh, config='--psm 8')  # Single word mode
#     return text.strip()

# def is_similar(plate1, plate2, pos_threshold=0.1 * frame_width, size_threshold=0.2):
#     """Check if two detections are likely the same plate based on position and size."""
#     pos1 = plate1['position']
#     pos2 = plate2['position']
#     dist = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
#     size1 = plate1['size']
#     size2 = plate2['size']
#     size_diff = abs(size1[0] - size2[0]) / size1[0] + abs(size1[1] - size2[1]) / size1[1]
#     return dist < pos_threshold and size_diff < size_threshold

# # Frame rate calculation variables
# frame_count = 0
# start_time = time.time()

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_count += 1
#     frame_start_time = time.time()
#     current_time = time.time()

#     # Perform detection with timing
#     detection_start = time.time()
#     boxes, scores, class_ids = yolov8_detector(frame)
#     detection_end = time.time()
#     DETECTION_LATENCY.observe(detection_end - detection_start)

#     # Process detections
#     for box, _, _ in zip(boxes, scores, class_ids):
#         x_min, y_min, x_max, y_max = map(int, box)
#         plate_region = frame[y_min:y_max, x_min:x_max]
#         license_plate_text = extract_license_plate_text(plate_region)

#         if not license_plate_text:
#             continue

#         # Calculate plate properties
#         center_x = (x_min + x_max) / 2
#         center_y = (y_min + y_max) / 2
#         width = x_max - x_min
#         height = y_max - y_min

#         # Check if this detection matches an active plate
#         matched = False
#         for plate_id, plate in active_plates.items():
#             if is_similar(plate, {'position': (center_x, center_y), 'size': (width, height)}):
#                 plate['texts'].append(license_plate_text)
#                 plate['last_seen'] = current_time
#                 plate['position'] = (center_x, center_y)
#                 plate['size'] = (width, height)
#                 matched = True
#                 DETECTIONS_TOTAL.inc()
#                 break

#         if not matched:
#             # New plate detected
#             active_plates[plate_counter] = {
#                 'texts': [license_plate_text],
#                 'last_seen': current_time,
#                 'position': (center_x, center_y),
#                 'size': (width, height)
#             }
#             DETECTIONS_TOTAL.inc()
#             plate_counter += 1

#     # Remove plates not seen for 10 seconds and process them
#     to_remove = [plate_id for plate_id, plate in active_plates.items() if current_time - plate['last_seen'] > 10]
#     detections = []

#     for plate_id in to_remove:
#         plate = active_plates[plate_id]
#         texts = plate['texts']
#         if texts:
#             # Select the most common text
#             most_common_text = Counter(texts).most_common(1)[0][0]
#             PLATES_PROCESSED.inc()
#             detections.append({
#                 "id": plate_id,
#                 "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(plate['last_seen'])),
#                 "license_plate_text": most_common_text
#             })
#         del active_plates[plate_id]

#     # Send detections if any
#     if detections:
#         json_output = {"detections": detections}
#         print("\nDetected Objects:")
#         print(json.dumps(json_output, indent=4))
#         producer.send('input-topic', value=json_output)

#     # Calculate frame rate
#     elapsed_time = time.time() - start_time
#     if elapsed_time > 1.0:
#         fps = frame_count / elapsed_time
#         FRAME_PROCESS_RATE.set(fps)
#         frame_count = 0
#         start_time = time.time()

#     # Visualize detections
#     combined_img = yolov8_detector.draw_detections(frame)
#     cv2.imshow("Detected Objects", combined_img)

#     # Quit on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()








# #test 


# import cv2
# import json
# import time
# from yolov8 import YOLOv8
# from kafka import KafkaProducer
# import easyocr
# import numpy as np
# from prometheus_client import Counter as PrometheusCounter, Histogram, Gauge, start_http_server
# from collections import Counter

# # Kafka Producer Configuration
# producer = KafkaProducer(
#     bootstrap_servers='localhost:9092',
#     value_serializer=lambda v: json.dumps(v).encode('utf-8')
# )

# # Prometheus Metrics
# start_http_server(8000)
# DETECTIONS_TOTAL = PrometheusCounter('yolo_detections_total', 'Total number of license plates detected')
# DETECTION_LATENCY = Histogram('yolo_detection_seconds', 'Time taken for YOLO detection')
# FRAME_PROCESS_RATE = Gauge('yolo_frame_process_rate', 'Frames processed per second')
# PLATES_PROCESSED = PrometheusCounter('yolo_plates_processed_total', 'Total number of unique plates processed')

# # Initialize video capture and OCR
# cap = cv2.VideoCapture("tc.mp4")
# frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# reader = easyocr.Reader(['en'])

# # Initialize YOLOv8 with ONNX model
# model_path = "mlmodel.onnx"
# yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

# # Define detection area
# area = [(1, 50), (1, 150), (1950, 150), (1950, 50)]

# cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

# # Detection tracking
# plate_counter = 1
# active_plates = {}
# processed_numbers = set()

# # Open file for writing car plate data
# with open("car_plate_data.txt", "a") as file:
#     file.write("NumberPlate\tDateTime\n")

# def extract_license_plate_text(plate_region):
#     """Extract text from detected license plate using EasyOCR with preprocessing."""
#     gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
#     bfilter = cv2.bilateralFilter(gray, 11, 11, 17)
#     ocr_result = reader.readtext(bfilter)
#     if ocr_result:
#         text = ocr_result[0][1].strip()
#         text = text.replace('(', '').replace(')', '').replace(',', '').replace(']','')
#         return text
#     return ""

# def is_similar(plate1, plate2, pos_threshold=0.1 * frame_width, size_threshold=0.2):
#     """Check if two detections are likely the same plate."""
#     pos1 = plate1['position']
#     pos2 = plate2['position']
#     dist = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
#     size1 = plate1['size']
#     size2 = plate2['size']
#     size_diff = abs(size1[0] - size2[0]) / size1[0] + abs(size1[1] - size2[1]) / size1[1]
#     return dist < pos_threshold and size_diff < size_threshold

# frame_count = 0
# start_time = time.time()

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_count += 1
#     current_time = time.time()

#     # Perform detection with timing
#     detection_start = time.time()
#     boxes, scores, class_ids = yolov8_detector(frame)
#     detection_end = time.time()
#     DETECTION_LATENCY.observe(detection_end - detection_start)

#     detections = []
    
#     # Process detections
#     for box, score, class_id in zip(boxes, scores, class_ids):
#         x_min, y_min, x_max, y_max = map(int, box)
        
#         # Calculate center point for area check
#         center_x = (x_min + x_max) / 2
#         center_y = (y_min + y_max) / 2
        
#         # Check if detection is within specified area
#         if cv2.pointPolygonTest(np.array(area, np.int32), (center_x, center_y), False) >= 0:
#             plate_region = frame[y_min:y_max, x_min:x_max]
#             license_plate_text = extract_license_plate_text(plate_region)

#             if not license_plate_text or license_plate_text in processed_numbers:
#                 continue

#             # Calculate plate properties
#             width = x_max - x_min
#             height = y_max - y_min

#             # Check for matching active plates
#             matched = False
#             for plate_id, plate in active_plates.items():
#                 if is_similar(plate, {'position': (center_x, center_y), 'size': (width, height)}):
#                     plate['texts'].append(license_plate_text)
#                     plate['last_seen'] = current_time
#                     plate['position'] = (center_x, center_y)
#                     plate['size'] = (width, height)
#                     matched = True
#                     DETECTIONS_TOTAL.inc()
#                     break

#             if not matched:
#                 active_plates[plate_counter] = {
#                     'texts': [license_plate_text],
#                     'last_seen': current_time,
#                     'position': (center_x, center_y),
#                     'size': (width, height)
#                 }
#                 DETECTIONS_TOTAL.inc()
                
#                 # Save to text file and prepare Kafka message
#                 processed_numbers.add(license_plate_text)
#                 current_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(current_time))
#                 with open("car_plate_data.txt", "a") as file:
#                     file.write(f"{license_plate_text}\t{current_datetime}\n")
                
#                 detections.append({
#                     "id": plate_counter,
#                     "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(current_time)),
#                     "license_plate_text": license_plate_text
#                 })
#                 plate_counter += 1
#                 PLATES_PROCESSED.inc()

#     # Remove expired plates
#     to_remove = [plate_id for plate_id, plate in active_plates.items() 
#                 if current_time - plate['last_seen'] > 10]
#     for plate_id in to_remove:
#         del active_plates[plate_id]

#     # Send detections to Kafka
#     if detections:
#         json_output = {"detections": detections}
#         print("\nDetected Objects:")
#         print(json.dumps(json_output, indent=4))
#         producer.send('input-topic', value=json_output)

#     # Calculate frame rate
#     elapsed_time = time.time() - start_time
#     if elapsed_time > 1.0:
#         fps = frame_count / elapsed_time
#         FRAME_PROCESS_RATE.set(fps)
#         frame_count = 0
#         start_time = time.time()

#     # Visualize detections with detection area
#     combined_img = yolov8_detector.draw_detections(frame)
#     cv2.polylines(combined_img, [np.array(area, np.int32)], True, (255, 0, 0), 2)  # Draw detection area
#     cv2.imshow("Detected Objects", combined_img)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()






# import cv2
# import json
# import time
# from yolov8 import YOLOv8
# from kafka import KafkaProducer
# import easyocr
# import numpy as np
# from prometheus_client import Counter as PrometheusCounter, Histogram, Gauge, start_http_server
# from collections import Counter

# # Kafka Producer Configuration
# producer = KafkaProducer(
#     bootstrap_servers='localhost:9092',
#     value_serializer=lambda v: json.dumps(v).encode('utf-8')
# )

# # Prometheus Metrics
# start_http_server(8000)
# DETECTIONS_TOTAL = PrometheusCounter('yolo_detections_total', 'Total number of license plates detected')
# DETECTION_LATENCY = Histogram('yolo_detection_seconds', 'Time taken for YOLO detection')
# FRAME_PROCESS_RATE = Gauge('yolo_frame_process_rate', 'Frames processed per second')
# PLATES_PROCESSED = PrometheusCounter('yolo_plates_processed_total', 'Total number of unique plates processed')

# # Initialize video capture and OCR
# cap = cv2.VideoCapture("tc.mp4")
# frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# reader = easyocr.Reader(['en'])

# # Initialize YOLOv8 with ONNX model
# model_path = "mlmodel.onnx"
# yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

# # Define detection area
# area = [(1, 50), (1, 150), (1950, 150), (1950, 50)]

# cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

# # Detection tracking
# plate_counter = 1
# active_plates = {}
# processed_numbers = set()

# # Open file for writing car plate data
# with open("car_plate_data.txt", "a") as file:
#     file.write("NumberPlate\tDateTime\n")

# def preprocess_image_1(image):
#     """First preprocessing method: Grayscale + Bilateral Filter"""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     processed = cv2.bilateralFilter(gray, 11, 17, 17)
#     return processed

# def preprocess_image_2(image):
#     """Second preprocessing method: Grayscale + CLAHE + Thresholding"""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(gray)
#     _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return binary

# def extract_license_plate_text(plate_region):
#     """Extract text from detected license plate using EasyOCR with dual preprocessing."""
#     try:
#         # First OCR attempt with preprocessing method 1
#         processed_1 = preprocess_image_1(plate_region)
#         ocr_result_1 = reader.readtext(processed_1)
#         plate_text_1 = " ".join([res[1] for res in ocr_result_1]).strip()
#         confidence_1 = max([res[2] for res in ocr_result_1], default=0)

#         # Second OCR attempt with preprocessing method 2
#         processed_2 = preprocess_image_2(plate_region)
#         ocr_result_2 = reader.readtext(processed_2)
#         plate_text_2 = " ".join([res[1] for res in ocr_result_2]).strip()
#         confidence_2 = max([res[2] for res in ocr_result_2], default=0)

#         # Double-check logic
#         final_plate_text = None
#         if plate_text_1 == plate_text_2 and plate_text_1:  # Exact match
#             final_plate_text = plate_text_1
#         elif confidence_1 > 0.8 and plate_text_1:  # High confidence in first result
#             final_plate_text = plate_text_1
#         elif confidence_2 > 0.8 and plate_text_2:  # High confidence in second result
#             final_plate_text = plate_text_2

#         if final_plate_text:
#             final_plate_text = final_plate_text.replace('(', '').replace(')', '').replace(',', '').replace(']','')
#             print(f"Verified Text: {final_plate_text} (Conf1: {confidence_1:.2f}, Conf2: {confidence_2:.2f})")
#             return final_plate_text
#         else:
#             print(f"OCR mismatch or low confidence: {plate_text_1} vs {plate_text_2}")
#             return ""

#     except Exception as e:
#         print(f"Error processing OCR: {str(e)}")
#         return ""

# def is_similar(plate1, plate2, pos_threshold=0.1 * frame_width, size_threshold=0.2):
#     """Check if two detections are likely the same plate."""
#     pos1 = plate1['position']
#     pos2 = plate2['position']
#     dist = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
#     size1 = plate1['size']
#     size2 = plate2['size']
#     size_diff = abs(size1[0] - size2[0]) / size1[0] + abs(size1[1] - size2[1]) / size1[1]
#     return dist < pos_threshold and size_diff < size_threshold

# frame_count = 0
# start_time = time.time()

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_count += 1
#     current_time = time.time()

#     # Perform detection with timing
#     detection_start = time.time()
#     boxes, scores, class_ids = yolov8_detector(frame)
#     detection_end = time.time()
#     DETECTION_LATENCY.observe(detection_end - detection_start)

#     detections = []
    
#     # Process detections
#     for box, score, class_id in zip(boxes, scores, class_ids):
#         x_min, y_min, x_max, y_max = map(int, box)
        
#         # Calculate center point for area check
#         center_x = (x_min + x_max) / 2
#         center_y = (y_min + y_max) / 2
        
#         # Check if detection is within specified area
#         if cv2.pointPolygonTest(np.array(area, np.int32), (center_x, center_y), False) >= 0:
#             plate_region = frame[y_min:y_max, x_min:x_max]
#             license_plate_text = extract_license_plate_text(plate_region)

#             if not license_plate_text or license_plate_text in processed_numbers:
#                 continue

#             # Calculate plate properties
#             width = x_max - x_min
#             height = y_max - y_min

#             # Check for matching active plates
#             matched = False
#             for plate_id, plate in active_plates.items():
#                 if is_similar(plate, {'position': (center_x, center_y), 'size': (width, height)}):
#                     plate['texts'].append(license_plate_text)
#                     plate['last_seen'] = current_time
#                     plate['position'] = (center_x, center_y)
#                     plate['size'] = (width, height)
#                     matched = True
#                     DETECTIONS_TOTAL.inc()
#                     break

#             if not matched:
#                 active_plates[plate_counter] = {
#                     'texts': [license_plate_text],
#                     'last_seen': current_time,
#                     'position': (center_x, center_y),
#                     'size': (width, height)
#                 }
#                 DETECTIONS_TOTAL.inc()
                
#                 # Save to text file and prepare Kafka message
#                 processed_numbers.add(license_plate_text)
#                 current_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(current_time))
#                 with open("car_plate_data.txt", "a") as file:
#                     file.write(f"{license_plate_text}\t{current_datetime}\n")
                
#                 detections.append({
#                     "id": plate_counter,
#                     "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(current_time)),
#                     "license_plate_text": license_plate_text
#                 })
#                 plate_counter += 1
#                 PLATES_PROCESSED.inc()

#     # Remove expired plates
#     to_remove = [plate_id for plate_id, plate in active_plates.items() 
#                 if current_time - plate['last_seen'] > 10]
#     for plate_id in to_remove:
#         del active_plates[plate_id]

#     # Send detections to Kafka
#     if detections:
#         json_output = {"detections": detections}
#         print("\nDetected Objects:")
#         print(json.dumps(json_output, indent=4))
#         producer.send('input-topic', value=json_output)

#     # Calculate frame rate
#     elapsed_time = time.time() - start_time
#     if elapsed_time > 1.0:
#         fps = frame_count / elapsed_time
#         FRAME_PROCESS_RATE.set(fps)
#         frame_count = 0
#         start_time = time.time()

#     # Visualize detections with detection area
#     combined_img = yolov8_detector.draw_detections(frame)
#     cv2.polylines(combined_img, [np.array(area, np.int32)], True, (255, 0, 0), 2)  # Draw detection area
#     cv2.imshow("Detected Objects", combined_img)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




# import cv2
# import json
# import time
# from yolov8 import YOLOv8
# from kafka import KafkaProducer
# import easyocr
# import numpy as np
# from prometheus_client import Counter as PrometheusCounter, Histogram, Gauge, start_http_server
# from datetime import datetime

# # Kafka Producer Configuration
# producer = KafkaProducer(
#     bootstrap_servers='localhost:9092',
#     value_serializer=lambda v: json.dumps(v).encode('utf-8')
# )

# # Prometheus Metrics
# start_http_server(8000)
# DETECTIONS_TOTAL = PrometheusCounter('yolo_detections_total', 'Total number of license plates detected')
# DETECTION_LATENCY = Histogram('yolo_detection_seconds', 'Time taken for YOLO detection')
# FRAME_PROCESS_RATE = Gauge('yolo_frame_process_rate', 'Frames processed per second')
# PLATES_PROCESSED = PrometheusCounter('yolo_plates_processed_total', 'Total number of unique plates processed')

# # Initialize video capture and OCR
# cap = cv2.VideoCapture("tc.mp4")
# frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# reader = easyocr.Reader(['en'])

# # Initialize YOLOv8 with ONNX model
# model_path = "mlmodel.onnx"
# yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

# # Define detection area and line
# area = [(1, 50), (1, 150), (1950, 150), (1950, 50)]
# detection_line_y = 430

# cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

# # Detection tracking
# plate_counter = 1
# processed_numbers = set()

# # Open file for writing car plate data
# with open("car_plate_data.txt", "a") as file:
#     file.write("NumberPlate\tDateTime\n")

# def preprocess_image_1(image):
#     """First preprocessing method: Grayscale + Bilateral Filter"""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     processed = cv2.bilateralFilter(gray, 11, 17, 17)
#     return processed

# def preprocess_image_2(image):
#     """Second preprocessing method: Grayscale + CLAHE + Thresholding"""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(gray)
#     _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return binary

# def extract_license_plate_text(plate_region):
#     """Extract text from detected license plate using EasyOCR with dual preprocessing."""
#     try:
#         # First OCR attempt with preprocessing method 1
#         processed_1 = preprocess_image_1(plate_region)
#         ocr_result_1 = reader.readtext(processed_1)
#         plate_text_1 = " ".join([res[1] for res in ocr_result_1]).strip()
#         confidence_1 = max([res[2] for res in ocr_result_1], default=0)

#         # Second OCR attempt with preprocessing method 2
#         processed_2 = preprocess_image_2(plate_region)
#         ocr_result_2 = reader.readtext(processed_2)
#         plate_text_2 = " ".join([res[1] for res in ocr_result_2]).strip()
#         confidence_2 = max([res[2] for res in ocr_result_2], default=0)

#         # Double-check logic
#         final_plate_text = None
#         if plate_text_1 == plate_text_2 and plate_text_1:  # Exact match
#             final_plate_text = plate_text_1
#         elif confidence_1 > 0.8 and plate_text_1:  # High confidence in first result
#             final_plate_text = plate_text_1
#         elif confidence_2 > 0.8 and plate_text_2:  # High confidence in second result
#             final_plate_text = plate_text_2

#         if final_plate_text:
#             final_plate_text = final_plate_text.replace('(', '').replace(')', '').replace(',', '').replace(']','')
#             print(f"Verified Text: {final_plate_text} (Conf1: {confidence_1:.2f}, Conf2: {confidence_2:.2f})")
#             return final_plate_text
#         else:
#             print(f"OCR mismatch or low confidence: {plate_text_1} vs {plate_text_2}")
#             return ""

#     except Exception as e:
#         print(f"Error processing OCR: {str(e)}")
#         return ""

# frame_count = 0
# start_time = time.time()

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_count += 1
#     current_time = time.time()

#     # Resize frame for consistency with first script
#     frame = cv2.resize(frame, (1020, 500))

#     # Perform detection with timing
#     detection_start = time.time()
#     boxes, scores, class_ids = yolov8_detector(frame)
#     detection_end = time.time()
#     DETECTION_LATENCY.observe(detection_end - detection_start)

#     detections = []

#     # Process detections
#     for box, score, class_id in zip(boxes, scores, class_ids):
#         x_min, y_min, x_max, y_max = map(int, box)
        
#         # Calculate center of the bounding box
#         center_x = (x_min + x_max) // 2
#         center_y = (y_min + y_max) // 2

#         # Check if the center crosses the detection line
#         if abs(center_y - detection_line_y) < 25:  # Within 25 pixels of the line
#             plate_region = frame[y_min:y_max, x_min:x_max]
#             license_plate_text = extract_license_plate_text(plate_region)

#             if not license_plate_text or license_plate_text in processed_numbers:
#                 continue

#             # Mark this number as processed
#             processed_numbers.add(license_plate_text)
#             DETECTIONS_TOTAL.inc()
#             PLATES_PROCESSED.inc()

#             # Get current timestamp
#             current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#             # Save to text file
#             with open("car_plate_data.txt", "a") as file:
#                 file.write(f"{license_plate_text}\t{current_datetime}\n")

#             # Prepare Kafka message
#             detections.append({
#                 "id": plate_counter,
#                 "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(current_time)),
#                 "license_plate_text": license_plate_text
#             })
#             plate_counter += 1

#             # Draw bounding box and text on frame
#             cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
#             cv2.putText(frame, license_plate_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Send detections to Kafka
#     if detections:
#         json_output = {"detections": detections}
#         print("\nDetected Objects:")
#         print(json.dumps(json_output, indent=4))
#         producer.send('input-topic', value=json_output)

#     # Calculate frame rate
#     elapsed_time = time.time() - start_time
#     if elapsed_time > 1.0:
#         fps = frame_count / elapsed_time
#         FRAME_PROCESS_RATE.set(fps)
#         frame_count = 0
#         start_time = time.time()

#     # Draw detection line and area
#     cv2.line(frame, (0, detection_line_y), (1020, detection_line_y), (255, 0, 0), 2)
#     cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)

#     # Display frame
#     cv2.imshow("Detected Objects", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




import cv2
import json
import time
from yolov8 import YOLOv8
from kafka import KafkaProducer
import easyocr
import numpy as np
from prometheus_client import Counter as PrometheusCounter, Histogram, Gauge, start_http_server
from datetime import datetime
from difflib import SequenceMatcher  # For string similarity

# Kafka Producer Configuration
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Prometheus Metrics
start_http_server(8000)
DETECTIONS_TOTAL = PrometheusCounter('yolo_detections_total', 'Total number of license plates detected')
DETECTION_LATENCY = Histogram('yolo_detection_seconds', 'Time taken for YOLO detection')
FRAME_PROCESS_RATE = Gauge('yolo_frame_process_rate', 'Frames processed per second')
PLATES_PROCESSED = PrometheusCounter('yolo_plates_processed_total', 'Total number of unique plates processed')

# Initialize video capture and OCR
cap = cv2.VideoCapture("tc.mp4")
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
reader = easyocr.Reader(['en'], verbose=False)


# Initialize YOLOv8 with ONNX model
model_path = "mlmodel2.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

# Define detection area and line
area = [(1, 50), (1, 150), (1950, 150), (1950, 50)]
detection_line_y = 430

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

# Detection tracking
plate_counter = 1
processed_plates = {}  # Store plates with timestamps for consolidation

# Open file for writing car plate data
with open("car_plate_data.txt", "a") as file:
    file.write("NumberPlate\tDateTime\n")

def preprocess_image_1(image):
    """First preprocessing method: Grayscale + Bilateral Filter"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed = cv2.bilateralFilter(gray, 11, 17, 17)
    return processed

def preprocess_image_2(image):
    """Second preprocessing method: Grayscale + CLAHE + Thresholding"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def normalize_plate_text(text):
    """Normalize plate text by removing spaces and special characters."""
    return ''.join(c for c in text.upper() if c.isalnum())

def is_similar_plate(text1, text2, threshold=0.85):
    """Check if two plate texts are similar using string similarity."""
    norm_text1 = normalize_plate_text(text1)
    norm_text2 = normalize_plate_text(text2)
    similarity = SequenceMatcher(None, norm_text1, norm_text2).ratio()
    return similarity >= threshold

def extract_license_plate_text(plate_region):
    """Extract text from detected license plate using EasyOCR with dual preprocessing."""
    try:
        
        # First OCR attempt 
        processed_1 = preprocess_image_1(plate_region)
        ocr_result_1 = reader.readtext(processed_1)
        plate_text_1 = " ".join([res[1] for res in ocr_result_1]).strip()
        confidence_1 = max([res[2] for res in ocr_result_1], default=0)

        # Second OCR attempt 
        processed_2 = preprocess_image_2(plate_region)
        ocr_result_2 = reader.readtext(processed_2)
        plate_text_2 = " ".join([res[1] for res in ocr_result_2]).strip()
        confidence_2 = max([res[2] for res in ocr_result_2], default=0)

        # Double-check logic
        final_plate_text = None
        if plate_text_1 == plate_text_2 and plate_text_1:  # Exact match
            final_plate_text = plate_text_1
        elif confidence_1 > 0.8 and plate_text_1:  # High confidence in first result
            final_plate_text = plate_text_1
        elif confidence_2 > 0.8 and plate_text_2:  # High confidence in second result
            final_plate_text = plate_text_2

        if final_plate_text:
            final_plate_text = final_plate_text.replace('(', '').replace(')', '').replace(',', '').replace(']','')
            return final_plate_text
        return ""

    except Exception:
        return ""

frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = time.time()

    # Resize frame for consistency
    frame = cv2.resize(frame, (1020, 500))

    # Perform detection with timing
    detection_start = time.time()
    boxes, scores, class_ids = yolov8_detector(frame)
    detection_end = time.time()
    DETECTION_LATENCY.observe(detection_end - detection_start)

    detections = []

    # Process detections
    for box, score, class_id in zip(boxes, scores, class_ids):
        x_min, y_min, x_max, y_max = map(int, box)
        
        # Calculate center of the bounding box
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        # Check if the center crosses the detection line
        if abs(center_y - detection_line_y) < 25:  # Within 25 pixels of the line
            plate_region = frame[y_min:y_max, x_min:x_max]
            license_plate_text = extract_license_plate_text(plate_region)

            if not license_plate_text:
                continue

            # Check for similarity with previously processed plates
            is_duplicate = False
            for processed_text, data in processed_plates.items():
                if is_similar_plate(license_plate_text, processed_text):
                    # Update the timestamp and consolidate
                    data['last_seen'] = current_time
                    data['readings'].append(license_plate_text)
                    is_duplicate = True
                    break

            if not is_duplicate:
                # New plate detected
                processed_plates[license_plate_text] = {
                    'last_seen': current_time,
                    'readings': [license_plate_text],
                    'processed': False
                }

    # Process plates that haven't been seen for a while (e.g., 5 seconds)
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    to_remove = []
    for plate_text, data in processed_plates.items():
        if current_time - data['last_seen'] > 5 and not data['processed']:  # 5-second buffer
            # Choose the most common reading
            final_plate_text = max(set(data['readings']), key=data['readings'].count)
            DETECTIONS_TOTAL.inc()
            PLATES_PROCESSED.inc()

            # Save to text file
            with open("car_plate_data.txt", "a") as file:
                file.write(f"{final_plate_text}\t{current_datetime}\n")

            # Prepare Kafka message
            detections.append({
                "id": plate_counter,
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(current_time)),
                "license_plate_text": final_plate_text
            })
            plate_counter += 1
            data['processed'] = True

            # Mark for removal after processing
            to_remove.append(plate_text)

    # Remove processed plates
    for plate in to_remove:
        del processed_plates[plate]

    # Send detections to Kafka and print
    if detections:
        json_output = {"detections": detections}
        print("Detected Objects:")
        print(json.dumps(json_output, indent=4))
        producer.send('input-topic', value=json_output)

    # Draw bounding boxes and text for visualization
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        center_y = (y_min + y_max) // 2
        if abs(center_y - detection_line_y) < 25:
            plate_region = frame[y_min:y_max, x_min:x_max]
            license_plate_text = extract_license_plate_text(plate_region)
            if license_plate_text:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                cv2.putText(frame, license_plate_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Calculate frame rate
    elapsed_time = time.time() - start_time
    if elapsed_time > 1.0:
        fps = frame_count / elapsed_time
        FRAME_PROCESS_RATE.set(fps)
        frame_count = 0
        start_time = time.time()

    # Draw detection line and area
    cv2.line(frame, (0, detection_line_y), (1020, detection_line_y), (255, 0, 0), 2)
    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)

    # Display frame
    cv2.imshow("Detected Objects", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






# import cv2
# import json
# import time
# from yolov8 import YOLOv8
# from kafka import KafkaProducer
# import easyocr
# import numpy as np
# from prometheus_client import Counter as PrometheusCounter, Histogram, Gauge, start_http_server
# from datetime import datetime
# from difflib import SequenceMatcher  # For string similarity

# # Kafka Producer Configuration
# producer = KafkaProducer(
#     bootstrap_servers='localhost:9092',
#     value_serializer=lambda v: json.dumps(v).encode('utf-8')
# )

# # Prometheus Metrics
# start_http_server(8000)
# DETECTIONS_TOTAL = PrometheusCounter('yolo_detections_total', 'Total number of license plates detected')
# DETECTION_LATENCY = Histogram('yolo_detection_seconds', 'Time taken for YOLO detection')
# FRAME_PROCESS_RATE = Gauge('yolo_frame_process_rate', 'Frames processed per second')
# PLATES_PROCESSED = PrometheusCounter('yolo_plates_processed_total', 'Total number of unique plates processed')

# # Initialize video capture and OCR
# cap = cv2.VideoCapture("test.mp4")
# frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# reader = easyocr.Reader(['en'], verbose=False)

# # Initialize YOLOv8 with ONNX model
# model_path = "mlmodel2.onnx"
# yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

# # Define detection area and vertical line (moved to the right side)
# area = [(870, 1), (970, 1), (970, 1080), (870, 1080)]  # Adjusted for right-side vertical context
# detection_line_x = 900  # Positioned near the right edge of the 1020-width frame

# cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

# # Detection tracking
# plate_counter = 1
# processed_plates = {}  # Store plates with timestamps for consolidation

# # Open file for writing car plate data
# with open("car_plate_data.txt", "a") as file:
#     file.write("NumberPlate\tDateTime\n")

# def preprocess_image_1(image):
#     """First preprocessing method: Grayscale + Bilateral Filter"""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     processed = cv2.bilateralFilter(gray, 11, 17, 17)
#     return processed

# def preprocess_image_2(image):
#     """Second preprocessing method: Grayscale + CLAHE + Thresholding"""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(gray)
#     _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return binary

# def normalize_plate_text(text):
#     """Normalize plate text by removing spaces and special characters."""
#     return ''.join(c for c in text.upper() if c.isalnum())

# def is_similar_plate(text1, text2, threshold=0.85):
#     """Check if two plate texts are similar using string similarity."""
#     norm_text1 = normalize_plate_text(text1)
#     norm_text2 = normalize_plate_text(text2)
#     similarity = SequenceMatcher(None, norm_text1, norm_text2).ratio()
#     return similarity >= threshold

# def extract_license_plate_text(plate_region):
#     """Extract text from detected license plate using EasyOCR with dual preprocessing."""
#     try:
#         # First OCR attempt 
#         processed_1 = preprocess_image_1(plate_region)
#         ocr_result_1 = reader.readtext(processed_1)
#         plate_text_1 = " ".join([res[1] for res in ocr_result_1]).strip()
#         confidence_1 = max([res[2] for res in ocr_result_1], default=0)

#         # Second OCR attempt 
#         processed_2 = preprocess_image_2(plate_region)
#         ocr_result_2 = reader.readtext(processed_2)
#         plate_text_2 = " ".join([res[1] for res in ocr_result_2]).strip()
#         confidence_2 = max([res[2] for res in ocr_result_2], default=0)

#         # Double-check logic
#         final_plate_text = None
#         if plate_text_1 == plate_text_2 and plate_text_1:  # Exact match
#             final_plate_text = plate_text_1
#         elif confidence_1 > 0.8 and plate_text_1:  # High confidence in first result
#             final_plate_text = plate_text_1
#         elif confidence_2 > 0.8 and plate_text_2:  # High confidence in second result
#             final_plate_text = plate_text_2

#         if final_plate_text:
#             final_plate_text = final_plate_text.replace('(', '').replace(')', '').replace(',', '').replace(']','')
#             return final_plate_text
#         return ""

#     except Exception:
#         return ""

# frame_count = 0
# start_time = time.time()

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_count += 1
#     current_time = time.time()

#     # Resize frame for consistency
#     frame = cv2.resize(frame, (1020, 500))

#     # Perform detection with timing
#     detection_start = time.time()
#     boxes, scores, class_ids = yolov8_detector(frame)
#     detection_end = time.time()
#     DETECTION_LATENCY.observe(detection_end - detection_start)

#     detections = []

#     # Process detections
#     for box, score, class_id in zip(boxes, scores, class_ids):
#         x_min, y_min, x_max, y_max = map(int, box)
        
#         # Calculate center of the bounding box
#         center_x = (x_min + x_max) // 2
#         center_y = (y_min + y_max) // 2

#         # Check if the center crosses the vertical detection line on the right
#         if abs(center_x - detection_line_x) < 25:  # Within 25 pixels of the vertical line
#             plate_region = frame[y_min:y_max, x_min:x_max]
#             license_plate_text = extract_license_plate_text(plate_region)

#             if not license_plate_text:
#                 continue

#             # Check for similarity with previously processed plates
#             is_duplicate = False
#             for processed_text, data in processed_plates.items():
#                 if is_similar_plate(license_plate_text, processed_text):
#                     # Update the timestamp and consolidate
#                     data['last_seen'] = current_time
#                     data['readings'].append(license_plate_text)
#                     is_duplicate = True
#                     break

#             if not is_duplicate:
#                 # New plate detected
#                 processed_plates[license_plate_text] = {
#                     'last_seen': current_time,
#                     'readings': [license_plate_text],
#                     'processed': False
#                 }

#     # Process plates that haven't been seen for a while (e.g., 5 seconds)
#     current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     to_remove = []
#     for plate_text, data in processed_plates.items():
#         if current_time - data['last_seen'] > 5 and not data['processed']:  # 5-second buffer
#             # Choose the most common reading
#             final_plate_text = max(set(data['readings']), key=data['readings'].count)
#             DETECTIONS_TOTAL.inc()
#             PLATES_PROCESSED.inc()

#             # Save to text file
#             with open("car_plate_data.txt", "a") as file:
#                 file.write(f"{final_plate_text}\t{current_datetime}\n")

#             # Prepare Kafka message
#             detections.append({
#                 "id": plate_counter,
#                 "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(current_time)),
#                 "license_plate_text": final_plate_text
#             })
#             plate_counter += 1
#             data['processed'] = True

#             # Mark for removal after processing
#             to_remove.append(plate_text)

#     # Remove processed plates
#     for plate in to_remove:
#         del processed_plates[plate]

#     # Send detections to Kafka and print
#     if detections:
#         json_output = {"detections": detections}
#         print("Detected Objects:")
#         print(json.dumps(json_output, indent=4))
#         producer.send('input-topic', value=json_output)

#     # Draw bounding boxes and text for visualization
#     for box in boxes:
#         x_min, y_min, x_max, y_max = map(int, box)
#         center_x = (x_min + x_max) // 2
#         if abs(center_x - detection_line_x) < 25:
#             plate_region = frame[y_min:y_max, x_min:x_max]
#             license_plate_text = extract_license_plate_text(plate_region)
#             if license_plate_text:
#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
#                 cv2.putText(frame, license_plate_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Calculate frame rate
#     elapsed_time = time.time() - start_time
#     if elapsed_time > 1.0:
#         fps = frame_count / elapsed_time
#         FRAME_PROCESS_RATE.set(fps)
#         frame_count = 0
#         start_time = time.time()

#     # Draw vertical detection line on the right and area
#     cv2.line(frame, (detection_line_x, 0), (detection_line_x, 500), (255, 0, 0), 2)  # Vertical line from top to bottom
#     cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)

#     # Display frame
#     cv2.imshow("Detected Objects", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()