import cv2

cap = cv2.VideoCapture(0)

# tracker = cv2.legacy.TrackerMOSSE_create()
tracker = cv2.legacy.TrackerCSRT_create()


ret, img = cap.read()

bbox = cv2.selectROI("Cam", img)

tracker.init(img, bbox)

def drawRect(img, bbox):
    
    x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0, 255), 1)

while True:
    
    timer = cv2.getTickCount()
    ret, img = cap.read()

    if not ret:
        break
    
    
    ret , bbox = tracker.update(img)    
    if ret:
        
        drawRect(img, bbox)
        
    else:
        cv2.putText(img, "Lost", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,40,255), 1)
    
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer) 
    cv2.putText(img, str(int(fps)), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 1)

    cv2.imshow("Cam", img)
    
    cv2.waitKey(1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
     
cap.release()
cv2.destroyAllWindows()