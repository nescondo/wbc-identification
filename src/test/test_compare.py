from ultralytics import YOLO

# Comparison b/w validation vs. test datasets (rule out overfitting)
modelTest = YOLO("../models/yolo11/runs/detect/yolo11_train_20pt/weights/best.pt")
modelVal = YOLO("../models/yolo11/runs/detect/yolo11_train_20pt/weights/best.pt")

resultsVal = modelVal.val(
    data="../datasets/bccd/data.yaml",
    split="val", # run model on validation imgs
    save=False
)

resultsTest = modelTest.val(
    data="../datasets/bccd/data.yaml",
    split="test", # run model on test imgs
    save=False
)

valMAP50 = resultsVal.box.map50
valMAP50_95= resultsVal.box.map

testMAP50 = resultsTest.box.map50
testMAP50_95= resultsTest.box.map

print(f"Val mAP@50:    {valMAP50:.4f}")
print(f"Val mAP@50-95: {valMAP50_95:.4f}\n")

print(f"Test mAP@50:    {testMAP50:.4f}")
print(f"Test mAP@50-95: {testMAP50_95:.4f}\n")

print(f"Val - Test mAP@50 Gap: {(valMAP50 - testMAP50) * 100:.4f}%")
print(f"Val - Test mAP@50-95 Gap: {(valMAP50_95 - testMAP50_95) * 100:.4f}%")