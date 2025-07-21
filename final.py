from ultralytics import YOLO

model = YOLO("best.pt")
model.predict(source=0, conf = 0.4, imgsz = 300, show= True)