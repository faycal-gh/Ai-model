import cv2
import numpy as np
import onnxruntime as ort

# === CONFIG ===
ONNX_MODEL_PATH = "C:\\Users\\msii\\Desktop\\Ai-model\\best.onnx"  # path to your model
INPUT_SIZE = 640  # must match what you used during training/export
CONF_THRESHOLD = 0.3

# === Load ONNX model ===
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)

# === Preprocess input ===
def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = img[:, :, ::-1]  # BGR to RGB
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0  # HWC to CHW and normalize
    return np.expand_dims(img, axis=0)

# === Postprocess output ===
def postprocess(outputs, original_shape):
    predictions = outputs[0][0]  # [num_detections, 85]
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    class_probs = predictions[:, 5:]
    class_ids = np.argmax(class_probs, axis=1)
    confidences = scores * np.max(class_probs, axis=1)

    results = []
    for box, conf, cls_id in zip(boxes, confidences, class_ids):
        if conf > CONF_THRESHOLD:
            x, y, w, h = box
            x1 = int((x - w / 2) * original_shape[1] / INPUT_SIZE)
            y1 = int((y - h / 2) * original_shape[0] / INPUT_SIZE)
            x2 = int((x + w / 2) * original_shape[1] / INPUT_SIZE)
            y2 = int((y + h / 2) * original_shape[0] / INPUT_SIZE)
            results.append((x1, y1, x2, y2, int(cls_id), float(conf)))
    return results

# === Open webcam ===
cap = cv2.VideoCapture(0)  # no CAP_V4L2 on Windows

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("✅ Webcam opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    input_tensor = preprocess(frame)
    outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_tensor})
    detections = postprocess(outputs, frame.shape)

    # Draw boxes
    for x1, y1, x2, y2, cls_id, conf in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Class {cls_id} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("YOLOv8 ONNX Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
