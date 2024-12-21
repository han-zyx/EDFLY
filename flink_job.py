import requests

# Define the path to your image
image_path = 'sample_img.jpg'  # Update with the correct path to your image

# Open the image in binary mode
with open(image_path, 'rb') as f:
    frame = f.read()

# Send the image to the YOLOv8 service
response = requests.post('http://localhost:8000/predict', files={'file': (image_path, frame)})

# Check if the response is valid and parse it
if response.status_code == 200:
    try:
        detections = response.json()  # Try parsing the response as JSON
        print(f"Detections: {detections}")
    except ValueError:
        print("Error: Response is not in JSON format or is empty")
        print(f"Response text: {response.text}")
else:
    print(f"Error: Received status code {response.status_code}")
    print(f"Response text: {response.text}")
