from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")

results = model.train(
    data="../datasets/wbc-mendeley", 
    epochs=100, 
    imgsz=224,
    device="mps",
    name="yolo26n-cls_train"
    )