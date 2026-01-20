import cv2
import numpy as np



cv2.namedWindow("HSV")

def dummy():
    pass

cv2.createTrackbar("Hue Min", "HSV", 0, 179, dummy)
cv2.createTrackbar("Hue Max", "HSV", 179, 179, dummy)
cv2.createTrackbar("Sat Min", "HSV", 0, 255, dummy)
cv2.createTrackbar("Sat Max", "HSV", 255, 255, dummy)
cv2.createTrackbar("Value Min", "HSV", 0, 255, dummy)
cv2.createTrackbar("Value Max", "HSV", 255, 255, dummy)


while True:
    
    img = cv2.imread("cv/tutorial/RealTimeColorDetection/image.png")
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    h_min = cv2.getTrackbarPos("Hue Min", "HSV")
    h_max = cv2.getTrackbarPos("Hue Max", "HSV")
    s_min = cv2.getTrackbarPos("Sat Min", "HSV")
    s_max = cv2.getTrackbarPos("Sat Max", "HSV")
    v_min = cv2.getTrackbarPos("Value Min", "HSV")
    v_max = cv2.getTrackbarPos("Value Max", "HSV")
    
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    
    mask = cv2.inRange(imgHsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    
    mask_to_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    hstack = np.hstack([img, mask_to_bgr, result])
    
    # cv2.namedWindow("mask" , cv2.WINDOW_NORMAL)
    # cv2.imshow("mask", mask)
    
    # cv2.namedWindow("output img" , cv2.WINDOW_NORMAL)
    # cv2.imshow("output img", result)
    
    # cv2.namedWindow("img" , cv2.WINDOW_NORMAL)
    # cv2.imshow("img", img)
    
    cv2.namedWindow("images", cv2.WINDOW_NORMAL)
    cv2.imshow("images", hstack)
    
    
    cv2.waitKey(1)
    
    