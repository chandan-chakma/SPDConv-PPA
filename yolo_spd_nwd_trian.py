from ultralytics import YOLO

# Load your new custom architecture
model = YOLO("yolov8-spdcon-nwd.yaml")

# Print summary to see if the P2 head and SPD layers are there
model.info()
