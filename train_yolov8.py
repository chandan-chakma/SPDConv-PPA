from ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld

if __name__=="__main__":


    # 使用YOLOv8.yamy文件搭建的模型训练
    # model = YOLO(r"D:\bilibili\model\ultralytics-main\ultralytics\cfg\models\v8\yolov8_my.yaml")  # build a new model from YAML
    # results = model.train(data=r'D:\bilibili\model\ultralytics-main\ultralytics\cfg\datasets\VOC_my.yaml',
    #                       epochs=100, imgsz=640, batch=4)
    #
    # # 加载已训练好的模型权重搭建模型训练 
    # model = YOLO(r'D:\bilibili\model\ultralytics-main\tests\yolov8n.pt')  # load a pretrained model (recommended for training)
    # results = model.train(data=r'D:\bilibili\model\ultralytics-main\ultralytics\cfg\datasets\VOC_my.yaml',
    #                       epochs=100, imgsz=640, batch=4)

    # 使用自己的YOLOv8.yamy文件搭建模型并加载预训练权重训练模型
    model = YOLO(r"ultralytics\cfg\models\v8\RIM.yaml")\
        .load('yolov8n.pt')  # build from YAML and transfer weights

    results = model.train(
        data=r'ultralytics\cfg\datasets\VisDrone.yaml',
        epochs=100, imgsz=640, batch=8, 
        #Optimizer settings
        optimizer='AdamW',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Augmentation settings
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0001,
        flipud=0.5,
        fliplr=0.5,
        bgr=0.0,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        
        # Loss settings
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Other settings
        close_mosaic=20,
        amp=True,
        multi_scale=True,
        patience=50,
        save=True,
        save_period=10,
        cache=False,
        device='0',
        workers=8,
        project='RIM_results',
        name='exp',
        exist_ok=False,
        pretrained=True,
        verbose=True,
        seed=0,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        val=True,
        plots=True,
        )