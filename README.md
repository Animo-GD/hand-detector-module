# hand-detector-module
A module I made to detect hand landmarks and return the position and id of each landmark
# To use 
```bash
pip install opencv-python mediapipe
```
```python
import mediapipe as mp
import cv2
import time
from hand_detector_module import hand_detector

# Make a new object
detector = new detector()
cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = detector.find_hand(frame)
        landmarks = detector.find_position(frame)
        if landmarks:
            print(landmarks[4])
cap.release()
cv2.destroyAllWindows()
```
------------------
# Landmark
![landmark](hand-landmark.png)
