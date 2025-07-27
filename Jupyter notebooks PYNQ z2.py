#==================== code without push button (500 frames) ====================#

from pynq.overlays.base import BaseOverlay
from pynq.lib.video import *
import cv2
import numpy as np
import time
import os
import sys

# 1. Load PYNQ base overlay
base = BaseOverlay("base.bit")

# 2. HDMI output configuration
Mode = VideoMode(640, 480, 24)
hdmi_out = base.video.hdmi_out
hdmi_out.configure(Mode, PIXEL_BGR)
hdmi_out.start()

# 3. USB Camera config
frame_in_w = 640
frame_in_h = 480
videoIn = cv2.VideoCapture(0)
videoIn.set(cv2.CAP_PROP_FRAME_WIDTH, frame_in_w)
videoIn.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_in_h)

if not videoIn.isOpened():
    raise RuntimeError("ERROR: Could not open USB camera")

print("✅ Camera is open")

# 4. Load fire cascade
FIRE_CASCADE_PATH = "/home/xilinx/jupyter_notebooks/base/faisal/Ai-model/Ai-model/cascade.xml"

if not os.path.exists(FIRE_CASCADE_PATH):
    print(f"❌ Cascade file not found: {FIRE_CASCADE_PATH}")
    sys.exit(1)

fire_cascade = cv2.CascadeClassifier(FIRE_CASCADE_PATH)

if fire_cascade.empty():
    print("❌ Failed to load cascade classifier.")
    sys.exit(1)

print("🔥 Fire detection model loaded.")

# 5. Detection loop
frame_count = 0
fire_count = 0
start = time.time()

while frame_count < 500:  # or change to `while True:` for unlimited
    ret, frame = videoIn.read()
    if not ret:
        print("❌ Failed to read frame.")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fire = fire_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in fire:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, '🔥 Fire', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        fire_count += 1

    # Send frame to HDMI
    outframe = hdmi_out.newframe()
    outframe[:] = frame
    hdmi_out.writeframe(outframe)

    frame_count += 1
    time.sleep(0.01)

end = time.time()

# 6. Cleanup
videoIn.release()
print("\n✅ Detection finished.")
print(f"Frames processed: {frame_count}")
print(f"Fire detections: {fire_count}")
print(f"FPS: {(frame_count)/(end - start):.2f}")


#==================== code with push button to top live detection ====================#


from pynq.overlays.base import BaseOverlay
from pynq.lib.video import *
import cv2
import numpy as np
import time
import os
import sys

# 1. Load PYNQ base overlay
base = BaseOverlay("base.bit")

# 2. Setup HDMI output
Mode = VideoMode(640, 480, 24)
hdmi_out = base.video.hdmi_out
hdmi_out.configure(Mode, PIXEL_BGR)
hdmi_out.start()

# 3. USB camera config
frame_in_w = 640
frame_in_h = 480
videoIn = cv2.VideoCapture(0)
videoIn.set(cv2.CAP_PROP_FRAME_WIDTH, frame_in_w)
videoIn.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_in_h)

if not videoIn.isOpened():
    raise RuntimeError("ERROR: Could not open USB camera")

print("✅ Camera opened")

# 4. Load fire cascade
FIRE_CASCADE_PATH = "/home/xilinx/jupyter_notebooks/base/faisal/Ai-model/Ai-model/cascade.xml"
if not os.path.exists(FIRE_CASCADE_PATH):
    print("❌ Fire cascade file not found.")
    sys.exit(1)

fire_cascade = cv2.CascadeClassifier(FIRE_CASCADE_PATH)
if fire_cascade.empty():
    print("❌ Failed to load cascade.")
    sys.exit(1)

print("🔥 Fire cascade loaded.")

# 5. Setup button (BTN0)
button0 = base.buttons[0]

# 6. Live loop
frame_count = 0
fire_count = 0
start = time.time()

print("🔴 Running fire detection... Press BTN0 to stop.")

while True:
    if button0.read():
        print("🛑 BTN0 pressed. Stopping...")
        break

    ret, frame = videoIn.read()
    if not ret:
        print("❌ Frame read failed.")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fire = fire_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in fire:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, '🔥 Fire', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        fire_count += 1

    outframe = hdmi_out.newframe()
    outframe[:] = frame
    hdmi_out.writeframe(outframe)

    frame_count += 1
    time.sleep(0.01)

end = time.time()

# 7. Cleanup
videoIn.release()
print("\n✅ Fire detection ended.")
print(f"🕒 FPS: {(frame_count)/(end - start):.2f}")
print(f"🔥 Fire detections: {fire_count}")
