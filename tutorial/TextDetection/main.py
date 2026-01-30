import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'D:\\Tess\\tesseract.exe'

img = cv2.imread("cv/tutorial/TextDetection/img.png")

imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

# print(pytesseract.image_to_string(img))

# Detecting Characters
# print(pytesseract.image_to_boxes(img))

# hImg, wImg, _ = img.shape

# boxes = pytesseract.image_to_boxes(img)

# for b in boxes.splitlines():
    
#     # print(b)
    
#     b = b.split(" ")
#     print(b)
    
#     x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])

#     cv2.rectangle(img, (x,hImg - y), (w,hImg-h), (0,0,255), 2)
#     cv2.putText(img,str(b[0]), (x,hImg - y), cv2.FONT_HERSHEY_PLAIN, 2,(0,0,0),2)
    
    
# print(pytesseract.image_to_data(img))
boxes = pytesseract.image_to_data(img)

for idx,b in enumerate(boxes.splitlines()):
    # print(b)
    b = b.split()
    print(b)
    if idx == 0:
        continue
    x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
    print(b[10])
    if len(b) > 11: 
        cv2.rectangle(img, (x,y), (w,y), (0,0,255), 2)
        cv2.putText(img,str(b[11]), (x,y), cv2.FONT_HERSHEY_PLAIN, 2,(0,0,0),2)
    
    
    
cv2.namedWindow("img", cv2.WINDOW_NORMAL)

cv2.imshow("img",img)

cv2.waitKey(0)