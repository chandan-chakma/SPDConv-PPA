from ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld

if __name__=="__main__":

        model = YOLO(r"runs\detect\train10\weights\best.pt") #load yolov8_smallhead model
        
        results = model.val(data=r'ultralytics\cfg\datasets\VisDrone.yaml',
                            split='test', imgsz=1024)
        print(results.box.map)
        print(results.box.map50)
        print(results.box.map75)