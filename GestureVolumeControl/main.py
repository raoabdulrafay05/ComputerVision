import cv2
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from CountFinger import HandTrackingModule as htm
from pycaw.pycaw import AudioUtilities

# -------------------- Setup --------------------
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Volume control variables
vol = 0
vol_bar = 140
smoothing = 2  # How smooth the volume changes
vol_prev = 0

# Audio device
device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume

# Hand detector
detector = htm.handDetector()

# -------------------- Main loop --------------------
while True:
    ret, img = cap.read()
    if not ret:
        break

    # Detect hands
    img, right, left = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Find distance between thumb and index finger
        length, img, x, y = detector.findTwoFingerPosition(4, 8, img)

        # Only update volume if distance is within a certain range
        if 55 < length < 160:
            vol_target = np.interp(length, [60, 155], [0, 1])

            # Smoothing: take average with previous volume
            vol = vol_prev + (vol_target - vol_prev) / smoothing
            vol_prev = vol

            # Set system volume
            volume.SetMasterVolumeLevelScalar(vol, None)

    # Draw volume bar
    vol_current = volume.GetMasterVolumeLevelScalar()
    vol_bar = np.interp(vol_current, [0, 1], [360, 140])
    cv2.rectangle(img, (70, 140), (110, 360), (0, 255, 0), 3)
    cv2.rectangle(img, (70, int(vol_bar)), (110, 360), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f"{int(vol_current*100)}%", (60, 400), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)

    # Display
    cv2.imshow("Volume Control", img)
    cv2.waitKey(1)

# -------------------- Cleanup --------------------
cap.release()
cv2.destroyAllWindows()
