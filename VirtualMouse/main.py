# Get the absolute path to the project root folder (cv)
import sys
import os
import cv2
import autopy
import time
import numpy as np
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from CountFinger import HandTrackingModule as hand

#################################
wCam, hCam = 640, 480
rec_side = 80
smoothing = 8
plocX = 0
plocY = 0
clocX , clocY = 0, 0

#################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = hand.handDetector(maxHands=1)

while True:
    ret, img = cap.read()
    
    if not ret :
        break
    
    img ,_,_= detector.findHands(img,draw = False)
    lmList = detector.findPosition(img, draw=False)
    # x_val , y_val = detector.findTwoFingerPosition(8,12,img, draw=True)
    
    if len(lmList) != 0:
        screen_w, screen_h = autopy.screen.size()
        
        cv2.rectangle(img, (rec_side,rec_side), (wCam - rec_side , hCam - 80 ), (255,0,255),2)
        
        if not(lmList[9][2] <= lmList[12][2] <= lmList[0][2])  and (not (lmList[5][2] <= lmList[8][2] <= lmList[0][2])): 
            length,img, x,y = detector.findTwoFingerPosition(8, 12, img)
            if length < 40:
                cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()
                time.sleep(0.1)

                


            
            
            
        
        if not (lmList[5][2] <= lmList[8][2] <= lmList[0][2]):
            
            x3 = np.interp(lmList[8][1], [rec_side, wCam - rec_side] , [0, screen_w])
            y3 = np.interp(lmList[8][2], [rec_side, hCam - 80], [0, screen_h])
            
            # Smoothining
            clocX = plocX + (x3 - plocX) / smoothing
            clocY = plocY + (y3 - plocY) / smoothing 
            
            if (rec_side + 1 <= lmList[8][1] <= wCam - rec_side -1 and rec_side +1 <= lmList[8][2] <= hCam - 80 -1):

                autopy.mouse.move(screen_w - clocX, clocY)

            cv2.circle(img, (lmList[8][1], lmList[8][2]), 15, (0, 255, 0), cv2.FILLED)
            plocX , plocY = clocX, clocY
            
            
    
    
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)    

    
    
    

cap.release()
cv2.destroyAllWindows()



