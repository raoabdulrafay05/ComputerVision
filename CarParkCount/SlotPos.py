import cv2
import pickle

width = 105

height = 42

try:
    with open("cv/CarParkCount/parkSlotPos", "rb") as f:
        posList = pickle.load(f)
except:
    posList = []

posList = []

img = cv2.imread("cv/CarParkCount/carParking.png")

def mouseClick(event, x, y, flags, params):
    
    if event == cv2.EVENT_LBUTTONDOWN:
        
        posList.append((x,y))
        
    if event == cv2.EVENT_RBUTTONDOWN:
        
        for i, pos in enumerate(posList):
            
            x1,y1 = pos
            
            if x1 < x < x1 + width and y1 < y < y1 + height:
                
                posList.pop(i)
                
    with open("cv/CarParkCount/parkSlotPos", "wb") as f:
        
        pickle.dump(posList, f)


while True:
    
    
    img = cv2.imread("cv/CarParkCount/carParking.png")

    # cv2.rectangle(img, (50, 100), (155, 140), (0,255,0), 1)
    
    for pos in posList:
        
        cv2.rectangle(img, (pos[0], pos[1]), (pos[0] + width, pos[1]+height), (0,255,0), 1)
    
    cv2.imshow("img", img)
    
    cv2.setMouseCallback("img", mouseClick)
    
    cv2.waitKey(1)