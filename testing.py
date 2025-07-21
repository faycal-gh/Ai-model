import cv2
import numpy as np
import time
import tflite_runtime.interpreter as tflite  # use 'import tensorflow.lite as tflite' if using full TensorFlow

import os
os.system("clear")
try: 
    interpreter = tflite.Interpreter(model_path="best_float32.tflite")
    interpreter.allocate_tensors()
except Exception as ex:
    print(ex)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# # Get input size
input_shape = input_details[0]['shape']
input_height = input_shape[1]
input_width = input_shape[2]

# # Video capture
video_capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not video_capture.isOpened():
    raise Exception("Could not open video device")

frame_count = 0
detect_every_n = 5
annotated_frame = None
prev_time = time.time()

def preprocess(frame):
    resized = cv2.resize(frame, (input_width, input_height))
    normalized = resized / 255.0  # normalize to [0,1]
    input_tensor = np.expand_dims(normalized, axis=0).astype(np.float32)
    return input_tensor

def postprocess(output_data, frame):
    print(f"Output data: {output_data}")
    print(f"Output data type: {type(output_data)}")
    print(f"Length of output_data: {len(output_data)}")


    h, w, _ = frame.shape
    boxes = output_data[0][0]  # Adjust according to model
    scores = output_data[1][0]
    classes = output_data[2][0]

    for i in range(len(scores)):
        if scores[i] > 0.6:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, top, right, bottom) = (int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h))
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{int(classes[i])} {scores[i]:.2f}", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return frame

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame_count += 1

    if frame_count % detect_every_n == 0:
        input_tensor = preprocess(frame)

        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()

        output_data = [interpreter.get_tensor(d['index']) for d in output_details]
        annotated_frame = postprocess(output_data, frame.copy())
    elif annotated_frame is None:
        annotated_frame = frame

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("YOLOv8 TFLite Inference", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()

