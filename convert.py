from ultralytics import YOLO

model = YOLO("mlmodel.pt")  # Adjust to your YOLOv8 model file path

model.export(format="onnx")  # Export to CoreML
