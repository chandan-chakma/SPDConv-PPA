from ultralytics.models import YOLO

if __name__ == "__main__":
    model = YOLO(r"ultralytics\cfg\models\v8\msp.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

    results = model.train(
        data=r"ultralytics\cfg\datasets\VisDrone.yaml", epochs=100, imgsz=640, batch=8, patience=50, verbose=True
    )
