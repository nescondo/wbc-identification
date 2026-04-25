from ultralytics import YOLO

# Comparison b/w validation vs. test datasets (rule out overfitting)

# Use when comparing object detection models
model = YOLO("../models/yolo26/runs/detect/yolo26_train/weights/best.pt")

resultsVal = model.val(
    data="../datasets/bccd/data.yaml", # or "../datasets/bccd/data.yaml" for object detection
    split="val", # run model on validation imgs,
    save_txt=True, # object detection info per img on (xywh, YOLO format) if you want to use
    project="cross_validation/yolo26",
    name="valid"
)

# use .val on both (.predict doesn't yield same metrics) -> make sure to split based on the test set (val, test, train), though
resultsTest = model.val(
    data="../datasets/bccd/data.yaml", # or "../datasets/bccd/data.yaml" for object detection
    split="test", # run model on test imgs
    save_txt=True,
    project="cross_validation/yolo26",
    name="test"
)

valMAP50 = resultsVal.box.map50
valMAP50_95= resultsVal.box.map

testMAP50 = resultsTest.box.map50
testMAP50_95 = resultsTest.box.map

print(f"Val mAP@50:    {valMAP50:.4f}")
print(f"Val mAP@50-95: {valMAP50_95:.4f}\n")

print(f"Test mAP@50:    {testMAP50:.4f}")
print(f"Test mAP@50-95: {testMAP50_95:.4f}\n")

print(f"Val - Test mAP@50 Gap: {(valMAP50 - testMAP50) * 100:.4f}%")
print(f"Val - Test mAP@50-95 Gap: {(valMAP50_95 - testMAP50_95) * 100:.4f}%")

# use when comparing classification models
# model = YOLO("../models/yolo26x-cls/runs/classify/yolo26x-cls_train/weights/best.pt")

# resultsVal = model.val(
#     data="../datasets/wbc-mendeley/", # or "../datasets/bccd/data.yaml" for object detection
#     split="val", # run model on validation imgs,
#     show=False,
#     save=False,
#     project="cross_validation/yolo26x-cls",
#     name="valid"
# )

# # have to use .val on both -> make sure to split based on the test set (val, test, train), though
# resultsTest = model.val(
#     source="../datasets/wbc-mendeley/", # or "../datasets/bccd/data.yaml" for object detection
#     split="test", # run model on test imgs
#     show=False,
#     save=False,
#     project="cross_validation/yolo26x-cls",
#     name="test"
# )

# valTop1 = resultsVal.top1
# valTop5 = resultsVal.top5

# testTop1 = resultsTest.top1
# testTop5 = resultsTest.top5

# print(f"Val Top 1:    {valTop1:.4f}")
# print(f"Val Top 5: {valTop5:.4f}\n")

# print(f"Test Top 1:    {testTop1:.4f}")
# print(f"Test Top 5: {testTop5:.4f}\n")

# print(f"Val - Test Top 1 Gap: {(valTop1 - testTop1) * 100:.4f}%")
# print(f"Val - Test Top 5 Gap: {(valTop5 - testTop5) * 100:.4f}%")