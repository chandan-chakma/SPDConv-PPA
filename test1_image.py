from ultralytics import YOLO

# Load a pretrained YOLOv8 model (e.g., yolov8n.pt for detection)
# Or load your custom trained model: model = YOLO('/path/to/your/best.pt')
model = YOLO(r'runs\detect\train10\weights\best.pt') 

# Perform inference on a single image
results = model.predict(source=r'datasets\Aerial Vehicles.v1i.yolov8\test\images\0000006_00159_d_0000001_jpg.rf.7606ebebd1510c7b066278526e4a875e.jpg', conf=0.25) 

# Process results (optional)
for r in results:
    # Print detection details
    print(r.boxes)  # Bounding boxes and their attributes
    print(r.probs)  # Class probabilities (for classification models)
    print(r.masks)  # Segmentation masks (for segmentation models)

    # Show the image with detections
    r.show() 

    # Save the image with detections
    r.save(filename='output_image.jpg')