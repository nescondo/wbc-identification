import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# generate predictions on test data (saved in test_predictions/model_name) AND crops WBCs (saved in wbc_crops)
model = YOLO("../models/yolo26_nms/runs/detect/yolo26_train_nms/weights/best.pt") # remember to update
class_names = model.names

model_name = "yolo26_nms" # remember to update
img_dir_str = "../datasets/bccd/test/images"
img_directory = os.fsencode(img_dir_str)

for img in os.listdir(img_directory):
    # save test predictions per img
    img_name = os.fsdecode(img)
    if img_name.endswith(".jpg"):
        img_stem = img_name.split("_jpg")[0] # split img name stem
        results = model.predict(
            source=f"{img_dir_str}/{img_name}", 
            show=True, 
            save=True,
            project="test_predictions",
            name="yolo26_nms", # remember to update
            exist_ok=True
        )
    
    # crop WBCs from individual images
    wbc_class = 2 # see src/datasets/bccd/data.yaml
    img = Image.open(f"{img_dir_str}/{img_name}")
    wbc_num = 0
    
    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.int().tolist(), result.boxes.conf.tolist()):
            class_name = model.names[cls]
            print(f"{class_name} ({conf:.2f}): {box}")

            if cls == wbc_class: 
                x1, y1, x2, y2 = map(int, box)
                crop = img.crop((x1, y1, x2, y2))
                crop_dir = Path(f"wbc_crops/{model_name}/{img_stem}")
                crop_dir.mkdir(parents=True, exist_ok=True) # create new dir, continue if exists
                crop.save(crop_dir / f"wbc_{wbc_num}.jpg")
                wbc_num += 1
            
# results = model.predict(
#     source="../datasets/bccd/test/images",
#     show=True,
#     save=True,
#     project="test_predictions",
#     name="yolo26_nms"
# )