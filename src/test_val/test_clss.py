from ultralytics import YOLO
from pathlib import Path

model_name = "yolo26"
model_suffix = "n-cls"

# recurse in directories within test
m_test_dir = Path("../datasets/wbc-mendeley/test")
m_test_imgs = list(m_test_dir.rglob("*.jpg"))

model = YOLO(f"../models/{model_name}{model_suffix}/runs/classify/{model_name}{model_suffix}_train/weights/best.pt")

# classifications on mendeley test set
results = model.predict(
    source=m_test_imgs,
    show=True,
    save=True,
    project=f"test_predictions/{model_name}{model_suffix}",
    name="mendeley"
)

# classifications on cropped imgs
results = model.predict(
    source=f"../datasets/wbc-crops/{model_name}",
    show=True,
    save=True,
    project=f"test_predictions/{model_name}{model_suffix}",
    name="wbc-crops"
)