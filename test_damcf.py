from ultralytics.models import YOLO

if __name__ == "__main__":
    model = YOLO(r"runs\detect\train23\weights\best.pt")  # load yolov8_smallhead model

    results = model.val(
        data=r"ultralytics/cfg/datasets/VisDrone.yaml",
        split="test",
        imgsz=640,
        batch=8,
        plots=True,
        conf=0.001,
        iou=0.6,
        max_det=300,
    )
    print(results.box.map)
    print(results.box.map50)
    print(results.box.map75)
