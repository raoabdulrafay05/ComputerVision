# import cv2
# import numpy as np
# import sys
# import os

# # Add project root to path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(project_root)
# from CountFinger import HandTrackingModule as htm

# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)

# detection = htm.handDetector()

# temp_arr = []
# rec_arr = []
# count = 0
# flag1, flag2 = False, False
# flag_Color = (0, 0, 255)
# cx, cy, w, h = 0,0,0,0
# x1,y1,x2,y2 = 0,0,0,0
# while True:
    
#     ret, img = cap.read()
#     if not ret:
#         break
    
#     img_ = detection.findHands(img)
    
#     lmList = detection.findPosition(img)
#     if len(lmList) != 0:
        
#         if 550 < lmList[8][1] < 600 and 50 < lmList[8][2] < 100:
#             flag1 = True
#             flag2 = True
#             flag_Color = (0, 255, 0)

#     cv2.rectangle(img, (550, 50), (600, 100), flag_Color, cv2.FILLED)
    
#     if flag1:
        
#         if len(lmList) != 0:
            
#             if not(lmList[5][2] <= lmList[8][2] <= lmList[0][2]) and not(lmList[17][2] <= lmList[20][2] <= lmList[0][2]):
#                 x1 = lmList[8][1]
#                 y1 = lmList[8][2]
#                 temp_arr.append(x1)
#                 temp_arr.append(y1)
#                 flag1 = False
                
                

#     if flag2 and not flag1:    
        
#         if len(lmList)!=0:
        
#             if not(lmList[9][2] <= lmList[12][2] <= lmList[0][2]) and not(lmList[5][2] <= lmList[8][2] <= lmList[0][2]):
#                 x2 = lmList[12][1]
#                 y2 = lmList[12][2]
#                 temp_arr.append(x2)
#                 temp_arr.append(y2)
#                 flag2 =  False

#     if len(temp_arr) == 4:
#         rec_arr.append(temp_arr)
#         print(temp_arr)
#         temp_arr = []
    
#     if rec_arr:
#         for arr in rec_arr:
#             cv2.rectangle(img, (arr[0], arr[1]), (arr[2], arr[3]), (255,0,255), cv2.FILLED)



#     cv2.imshow("img", img)
#     cv2.waitKey(1)
    
    
    
# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from CountFinger import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

smoothing = 1.2

class RectangleDrawer:
    def __init__(self):
        self.temp_arr = []
        self.rec_arr = []
        self.center_box = []
        self.count = 0

        self.flag1 = False
        self.flag2 = False
        self.flag_Color = (0, 0, 255)

        self.x1 = self.y1 = self.x2 = self.y2 = 0

    def update(self, lmList):
        """
        Updates temp_arr and rec_arr based on hand landmarks
        (Logic unchanged from original code)
        """
        has_hand = len(lmList) != 0

        if has_hand and self.count <= 5:
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
                self.center_box.append([(self.x1 + self.x2)//2,(self.y1+self.y2)//2, abs(self.x2-self.x1)//2, abs(self.y2-self.y1)//2])
                print(self.center_box)
                self.flag2 = False

        if len(self.temp_arr) == 4:
            self.rec_arr.append(self.temp_arr)
            # print(self.temp_arr)
            self.temp_arr = []
            self.flag_Color = (0,0,255)
            self.count += 1


# ------------------ MAIN ------------------



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
        # print("raw pts:", arr[0:5])
        x_min, y_min = min(arr[0],arr[2]), min(arr[1], arr[3])
        x_max, y_max = max(arr[0],arr[2]), max(arr[1], arr[3])
        # print("draw:", x_min, y_min, x_max, y_max)


        cv2.rectangle(img, (int(arr[0]), int(arr[1])), (int(arr[2]), int(arr[3])), (255, 0, 255), cv2.FILLED)
        
    if not(drawer.flag1 and drawer.flag2):
        if len(lmList) > 8 and lmList[17][2] <= lmList[20][2] <= lmList[0][2]:
            for idx, (arr, rec) in enumerate(zip(drawer.center_box, drawer.rec_arr)):
                if rec[0] < lmList[8][1] < rec[2] and rec[1] < lmList[8][2] < rec[3]:
                    cx,cy,w,h = arr
                    target_x = lmList[8][1]
                    target_y = lmList[8][2]
                    new_cx = cx + (target_x-cx)/smoothing
                    new_cy = cy + (target_y -cy)/smoothing
                    
                    rec[0] = new_cx - w
                    rec[1] = new_cy - h
                    rec[2] = new_cx + w
                    rec[3] = new_cy + h

                    drawer.center_box[idx][0] = new_cx
                    drawer.center_box[idx][1] = new_cy
                
                

    cv2.imshow("img", img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
