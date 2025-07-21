from ultralytics import YOLO

model = YOLO("best (2).pt") 
# model.predict(source="C:\\Users\\msii\\Desktop\\project\\src\\components\\image.png", imgsz=640, conf=0.6, save=True)
model.predict(source=0, imgsz=288, conf=0.6, show=True)