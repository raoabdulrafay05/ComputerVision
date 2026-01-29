import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'D:\\Tess\\tesseract.exe'

img = cv2.imread("cv/tutorial/TextDetection/img.png")

imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

# print(pytesseract.image_to_string(img))

# Detecting Characters
print(pytesseract.image_to_boxes(img))

hImg, wImg, _ = img.shape

boxes = pytesseract.image_to_boxes(img)

for b in boxes.splitlines():
    
    print(b)
    
    b = b.split(" ")
    print(b)
    
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])

    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    

cv2.namedWindow("img", cv2.WINDOW_NORMAL)

cv2.imshow("img",img)

cv2.waitKey(0)