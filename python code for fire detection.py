# -*- coding: utf-8 -*-
"""
Fire Detection using Cascade Classifier (No Arduino)
"""

import numpy as np
import cv2
import time

# Load your trained cascade classifier
fire_cascade = cv2.CascadeClassifier('cascade.xml')  # Ensure this file is in the same folder

# Start capturing from your webcam (Windows camera)
cap = cv2.VideoCapture(0)

# Optional: check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

count = 0

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Grayscale is standard for Haar cascades

    # Run fire detection on the frame
    fire = fire_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in fire:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        print('🔥 Fire detected! Frame count:', count)
        count += 1
        time.sleep(0.2)  # Slight delay to avoid overwhelming prints

    # Display the result
    cv2.imshow('Fire Detection', img)

    # Exit if ESC key is pressed
    if cv2.waitKey(100) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
