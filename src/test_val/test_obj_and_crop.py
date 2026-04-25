import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# generate predictions on test data (saved in test_predictions/model_name) AND crops WBCs (saved in wbc_crops)
model_name = "yolo26" # remember to update
img_dir_str = "../datasets/bccd/test/images"
img_directory = os.fsencode(img_dir_str)

model = YOLO(f"../models/{model_name}/runs/detect/{model_name}_train/weights/best.pt")
class_names = model.names

wbc_num = 0
for img in os.listdir(img_directory):
    # save test predictions per img
    img_name = os.fsdecode(img)
    if img_name.endswith(".jpg"):
        results = model.predict(
            source=f"{img_dir_str}/{img_name}", 
            show=True, 
            save=True,
            project="test_predictions",
            name=f"{model_name}",
            exist_ok=True
        )
    
        # crop WBCs from individual images
        wbc_class = 2 # see src/datasets/bccd/data.yaml
        img_pil = Image.open(f"{img_dir_str}/{img_name}")
        
        for result in results:
            for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.int().tolist(), result.boxes.conf.tolist()):
                class_name = model.names[cls]
                print(f"{class_name} ({conf:.2f}): {box}")

                if cls == wbc_class: 
                    x1, y1, x2, y2 = map(int, box)
                    crop = img_pil.crop((x1, y1, x2, y2))
                    crop_dir = Path(f"../datasets/wbc-crops/{model_name}/")
                    crop_dir.mkdir(parents=True, exist_ok=True) # create new dir, continue if exists
                    crop.save(crop_dir / f"wbc_{wbc_num}.jpg")
                    wbc_num += 1