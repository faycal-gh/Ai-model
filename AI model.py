import cv2
from ultralytics import YOLO
import time
model = YOLO("best.pt")  
video_capture = cv2.VideoCapture(0, cv2.CAP_V4L2)  
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
if not video_capture.isOpened():
    raise Exception("Could not open video device")
frame_count = 0
detect_every_n = 5  
annotated_frame = None
prev_time = time.time()
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame.")
        break
    frame_count += 1
    if frame_count % detect_every_n == 0:
        results = model(frame, imgsz=320, conf=0.6)
        annotated_frame = results[0].plot()
    elif annotated_frame is None:
        annotated_frame = frame 
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("YOLOv8 on Raspberry Pi", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
