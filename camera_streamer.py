from kafka import KafkaProducer
import cv2
import base64

# Initialize Kafka producer
producer = KafkaProducer(bootstrap_servers='localhost:9092')

# Open the camera (default camera = 0; replace with RTSP URL for an IP camera)
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to stop streaming.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Display the frame in a window
    cv2.imshow("Camera Feed", frame)
    
    # Encode the frame as a JPEG for streaming
    _, buffer = cv2.imencode('.jpg', frame)
    frame_data = base64.b64encode(buffer).decode('utf-8')
    
    # Send the encoded frame to the Kafka topic
    producer.send('camera-stream', value=frame_data.encode('utf-8'))
    
    # Quit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
