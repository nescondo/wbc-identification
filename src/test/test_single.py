from ultralytics import YOLO

# Extract object detection data: bounding box (xyxy - min coord, max coord, xywh - center pt., width, height format), class index, confidence score - DOES NOT SAVE - see test_model.py
model = YOLO("../models/yolo11/runs/detect/yolo11_train_20pt/weights/best.pt")
class_names = model.names
results = model.predict(
            source="../datasets/bccd/test/images/BloodImage_00038_jpg.rf.5f471a23f980c08d21a0ab1f91bafd46.jpg", 
            show=True, 
            save=False,
        )

for result in results:
    bbox_list = result.boxes.xyxy.tolist()  # bounding box of all objs
    clss_list = result.boxes.cls.int().tolist() # class index of all objs
    conf_list = result.boxes.conf.tolist() # confidence list of all objs
    for box, cls, conf in zip(bbox_list, clss_list, conf_list):
        print(f"Bounding box: {box}, Class index: {cls}, Class name: {class_names[cls]}, Confidence: {conf}")