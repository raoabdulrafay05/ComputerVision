import cv2
import time
import sys
import os
import numpy as np
import math
import random

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from CountFinger import HandTrackingModule as htm

handTracker = htm.handDetector(maxHands=1)

score = 0

cap = cv2.VideoCapture(0)

cap.set(3,640)
cap.set(4, 480)

x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

coff = np.polyfit(x, y, 2)

target_x, target_y = random.randint(50, 500), random.randint(50, 400)
color = (255,0,255)
counter = 0
watch = 1
timeStart = time.time()
totaltime = 30
while True:
    
    current = time.time()
        
    ret, img = cap.read()
    
    img = cv2.flip(img, 1)
    
    if not ret:
        break
    
    if watch:
        img, _, _ = handTracker.findHands(img)
        
        lmList = handTracker.findPosition(img)
        
        if lmList:
            
            raw_distance = math.sqrt(math.pow(lmList[5][1] - lmList[17][1],2)+math.pow(lmList[5][2] - lmList[17][2], 2))
            x1,y1 = lmList[17][1], lmList[17][2]
            x2,y2 = lmList[2][1], lmList[2][2]
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            A, B, C = coff
            
            distance = int(A*(raw_distance**2) + (B*raw_distance) + C)
            # print(distance)
            
        
            if distance <= 22:
                if min(x1, x2) < target_x < max(x1, x2) and min(y1, y2) < target_y < max(y1, y2):
                    color = (0,255,0)
                    counter = 1
                
            if  counter:
                counter +=1
                color = (0,255,0)
                
                if  counter == 3:
                    counter = 0
                    color = (255,0,255)
                    score +=1
                    target_x, target_y = random.randint(50, 550), random.randint(50, 400)
                    
        print(target_x, target_y)
        cv2.circle(img, (target_x, target_y), 25, (255,255,255),2)
        cv2.circle(img, (target_x, target_y), 25, color,cv2.FILLED)
        cv2.circle(img, (target_x, target_y), 8, (50,50,50),5)
        cv2.circle(img, (target_x, target_y), 3, (0,0,255),5)
        watch = int(totaltime - (time.time() - timeStart))
        cv2.putText(img, f'Time: {watch}' , (40, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (25,25,25), 1)
        cv2.putText(img, f'Score: {score}' , (400, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (15,150,15), 1)
        
    else:
        
        cv2.putText(img, "Game Over", (140, 220), cv2.FONT_HERSHEY_DUPLEX, 2,(255,255,255), 5)
        cv2.putText(img, "Game Over", (140, 220), cv2.FONT_HERSHEY_DUPLEX, 2,(0,0,0), 3)
        cv2.putText(img, f"  Score: {score}", (140, 280), cv2.FONT_HERSHEY_DUPLEX, 2,(255,255,255), 2)        
        cv2.putText(img, f"  Score: {score}", (140, 280), cv2.FONT_HERSHEY_DUPLEX, 2,(0,0,0), 1)
        cv2.putText(img, f"Press R to Restart", (110, 320), cv2.FONT_HERSHEY_DUPLEX, 2,(255,255,255), 2)        
        cv2.putText(img, f"Press R to Restart", (110, 320), cv2.FONT_HERSHEY_DUPLEX, 2,(0,0,0), 1)
        
        if cv2.waitKey(1) & 0xFF == ord('r'):
            score = 0
            watch = 1
            timeStart = time.time()
    
    
    cv2.imshow("img" , img)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()