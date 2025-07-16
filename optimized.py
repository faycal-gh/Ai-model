import cv2
import numpy as np
import onnxruntime as ort

# Load ONNX model
ort_session = ort.InferenceSession("best.onnx")

# Set input size
input_size = 416

def preprocess(frame):
    img = cv2.resize(frame, (input_size, input_size))
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess(frame)
    outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_tensor})

    # Optional: visualize or process detections here (more complex)
    cv2.imshow("ONNX Inference", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
