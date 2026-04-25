from ultralytics import YOLO

# Load a pre-trained model (COCO)
model = YOLO("yolo26n.pt")

# Train the model
results = model.train(data="../../datasets/bccd/data.yaml", 
                      epochs=50,
                      patience=20, 
                      imgsz=640, 
                      device="mps",
                    #   amp=False,
                      name="yolo26_train")