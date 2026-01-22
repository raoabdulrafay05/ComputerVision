import cv2

cap = cv2.VideoCapture(0)

tracker = cv2.legacy.TrackerMOSSE_create()

ret, img = cap.read()

bbox = cv2.selectROI("Cam", img)

tracker.init(img, bbox)

while True:
    
    timer = cv2.getTickCount()
    ret, img = cap.read()
    
    if not ret:
        break
    
    
    ret , bbox = tracker.update(img)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer) 
    
    if ret:
        
        drawRect()
        
    else:
        cv2.putText()
    
    cv2.putText(img, str(fps), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0), 2)
    cv2.imshow("Cam", img)
    
    cv2.waitKey(1)
    
cap.release()
cv2.destroyAllWindows()