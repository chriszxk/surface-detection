from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.pt")  # load an official model
model = YOLO("/home/xunkuai/crack_detection/v11-autodlv2/hk/hrnethead256-512-btl-ds/train/weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")