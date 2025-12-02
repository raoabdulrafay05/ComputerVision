import cv2
import time
import os
import CountFinger.HandTrackingModule as html

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "FingerImages"
mylist = os.listdir(folderPath)
print(mylist)
overlayList = []

for imPath in mylist:
    imgage = cv2.imread(f"{folderPath}/{imPath}")

p_time = 0

while True:
    
    res, img = cap.read()
    
    if not res:
        break
    
    ctime = time.time()
    fps = 1/(ctime-p_time)
    p_time = ctime
    
    cv2.putText(img, fps, (70, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,255,0),2)
    
    cv2.imshow("image", img)
    cv2.waitKey(1)    
