# from flask import Flask, request, jsonify, send_file
# from ultralytics import YOLO
# import cv2
# import numpy as np
# from io import BytesIO
# import os

# # Initialize the YOLO model (make sure the path to your fine-tuned model is correct)
# model = YOLO('mlmodel.pt')  # Replace with your fine-tuned model path
# print("Model loaded successfully.")

# app = Flask(__name__)

# # Serve static files (frontend HTML)
# @app.route('/')
# def index():
#     return app.send_static_file('index.html')

# # Predict route to handle image upload and return processed image
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get the file from the request
#         file = request.files['file']
        
#         # Read and decode the image
#         file_data = file.read()
#         img = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
        
#         if img is None:
#             return jsonify({"error": "Image decoding failed"}), 400

#         print(f"Received image with shape: {img.shape}")  # Debug: print image shape

#         # Perform inference using YOLO
#         results = model(img)  # Inference using the YOLO model

#         # Plot the results on the image (draw bounding boxes on the image)
#         result_img = results[0].plot()  # This overlays the bounding boxes and labels onto the image

#         # Convert the image with bounding boxes to a byte array to send back
#         _, img_encoded = cv2.imencode('.jpg', result_img)
#         img_bytes = img_encoded.tobytes()

#         return send_file(BytesIO(img_bytes), mimetype='image/jpeg')

#     except Exception as e:
#         print(f"Error: {e}")  # Log the error for debugging
#         return jsonify({"error": str(e)}), 500

# # Start the Flask app
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080)  # Expose the server on all IP addresses





import cv2
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
from flask_socketio import SocketIO

# Initialize the Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Load the YOLOv8 model
model = YOLO("mlmodel.pt")  # Adjust to your YOLOv8 model file path

# Global variable to store the camera object
cap = None
camera_index = 0  # Default camera index

# Initialize the camera capture
def init_camera(index=0):
    global cap
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Camera {index} failed to open.")
    return cap

# Function to capture and process frames
def gen():
    global cap
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)

        # Plot the results on the frame
        result_img = results[0].plot()

        # Convert the image to JPEG
        ret, jpeg = cv2.imencode('.jpg', result_img)
        if not ret:
            break

        # Yield the frame as multipart JPEG for streaming
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

# Flask route to start the camera and stream the feed
@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask route to get available cameras
@app.route('/available_cameras')
def available_cameras():
    cameras = []
    for index in range(5):  # Test up to 5 camera devices
        cap_test = cv2.VideoCapture(index)
        if cap_test.isOpened():
            cameras.append(f"Camera {index}")
        cap_test.release()
    return jsonify(cameras=cameras)

# Route to select the camera
@app.route('/set_camera', methods=['POST'])
def set_camera():
    global cap, camera_index
    camera_index = int(request.form['camera_index'])
    if cap:
        cap.release()  # Release the current camera if it exists
    init_camera(camera_index)
    return jsonify({"message": f"Camera {camera_index} selected successfully!"})

# Route to stop the camera feed
@app.route('/close_camera', methods=['POST'])
def close_camera():
    global cap
    if cap:
        cap.release()
        cap = None
    return jsonify({"message": "Camera closed successfully!"})

# Home route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
