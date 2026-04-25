from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt") # load a pretrained model (COCO)

# Train the model
results = model.train(data="../../datasets/bccd/data.yaml", 
                      epochs=50, # yolo11, yolo26 performance stalls at epoch ~35-45
                      patience=20, # wait 20 epochs before stopping run if performance stalls (prevents overfitting)
                      imgsz=640, 
                      device="mps", 
                      end2end=False,
                      name="yolo26_train_nms")