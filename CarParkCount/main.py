import cv2
import cvzone
import pickle
import numpy as np


cap = cv2.VideoCapture("cv/CarParkCount/carPark.mp4")

posList=[]
with open("cv/CarParkCount/parkSlotPos", "rb") as f:
    
    posList = pickle.load(f)
    
width, height = 105, 42

def checkParkingSlot(imgProcess):
    
    for pos in posList:
        x,y = pos

        imgCrop = imgProcess[y:y+height, x:x+width]
        
        # cv2.imshow(str(x*y), imgCrop)
        
        count = cv2.countNonZero(imgCrop)
        
        if count < 1100:
            
            color = (0,255,0)
            thickness = 5
            
        else:
            
            color = (0,0,255)
            thickness = 2
            
        cv2.rectangle(img, (pos[0], pos[1]), (pos[0] + width, pos[1]+height), color, thickness)
        cvzone.putTextRect(img, str(count), (x, y+height-2), 1, 1, offset=0)



while True:
    
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    ret, img = cap.read()
    
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    imgBlur = cv2.GaussianBlur(imgGray, (3,3), 1)
    
    imgThresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 29,13)
    
    # imgMedian = cv2.medianBlur(imgThresh, 5)    # -> Not Using Median
    
    kernel = np.ones((3,3), np.uint8)
    imgDilect = cv2.dilate(imgThresh,kernel, iterations=1)
        
    checkParkingSlot(imgDilect)
    
    cv2.namedWindow("image",cv2.WINDOW_NORMAL)
    cv2.imshow("image", img)
    
    # cv2.namedWindow("imgGray",cv2.WINDOW_NORMAL)
    # cv2.imshow("imgGray", imgBlur)
    
    # cv2.namedWindow("imgThresh",cv2.WINDOW_NORMAL)
    # cv2.imshow("imgThresh", imgThresh)
    
    # # cv2.namedWindow("imgMedian",cv2.WINDOW_NORMAL)
    # # cv2.imshow("imgMedian", imgMedian)
    
    # cv2.namedWindow("imgDilect",cv2.WINDOW_NORMAL)
    # cv2.imshow("imgDilect", imgDilect)
    
    

    
    cv2.waitKey(1)