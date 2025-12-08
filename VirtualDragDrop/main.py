import cv2
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from CountFinger import HandTrackingModule as htm


class RectangleDrawer:
    def __init__(self):
        self.temp_arr = []
        self.rec_arr = []
        self.count = 0

        self.flag1 = False
        self.flag2 = False
        self.flag_Color = (0, 0, 255)

        self.cx = self.cy = self.w = self.h = 0
        self.x1 = self.y1 = self.x2 = self.y2 = 0

    def update(self, lmList):
        """
        Updates temp_arr and rec_arr based on hand landmarks
        (Logic unchanged from original code)
        """
        has_hand = len(lmList) != 0

        if has_hand:
            ix, iy = lmList[8][1], lmList[8][2]

            if 550 < ix < 600 and 50 < iy < 100:
                self.flag1 = True
                self.flag2 = True
                self.flag_Color = (0, 255, 0)

        if self.flag1 and has_hand:
            if not (lmList[5][2] <= lmList[8][2] <= lmList[0][2]) and \
               not (lmList[17][2] <= lmList[20][2] <= lmList[0][2]):

                self.x1 = lmList[8][1]
                self.y1 = lmList[8][2]
                self.temp_arr.extend([self.x1, self.y1])
                self.flag1 = False

        if self.flag2 and not self.flag1 and has_hand:
            if not (lmList[9][2] <= lmList[12][2] <= lmList[0][2]) and \
               not (lmList[5][2] <= lmList[8][2] <= lmList[0][2]):
                

                self.x2 = lmList[12][1]
                self.y2 = lmList[12][2]
                self.temp_arr.extend([self.x2, self.y2])
                self.flag2 = False

        if len(self.temp_arr) == 4:
            self.rec_arr.append(self.temp_arr)
            print(self.temp_arr)
            self.temp_arr = []


# ------------------ MAIN ------------------

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detection = htm.handDetector()
drawer = RectangleDrawer()

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    if not ret:
        break

    detection.findHands(img)
    lmList = detection.findPosition(img)

    drawer.update(lmList)

    cv2.rectangle(img, (550, 50), (600, 100), drawer.flag_Color, cv2.FILLED)

    for arr in drawer.rec_arr:
        print("raw pts:", arr[0:5])
        x_min, y_min = min(arr[0],arr[2]), min(arr[1], arr[3])
        x_max, y_max = max(arr[0],arr[2]), max(arr[1], arr[3])
        print("draw:", x_min, y_min, x_max, y_max)


        cv2.rectangle(img, (int(arr[0]), int(arr[1])), (int(arr[2]), int(arr[3])), (255, 0, 255), cv2.FILLED)

    cv2.imshow("img", img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
