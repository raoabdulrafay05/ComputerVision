import cv2

# img = cv2.imread("cv/ObjectDetection/im.jpg")

classFile = "cv/ObjectDetection/coco.names"

configPath = "cv/ObjectDetection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

weightsPath = "cv/ObjectDetection/frozen_inference_graph.pb"

with open(classFile, "rt") as f:
    
    classNames = f.read().rstrip("\n").split("\n")
    

net = cv2.dnn_DetectionModel(weightsPath, configPath) 
        
net.setInputSize(320,320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

cap = cv2.VideoCapture(0)

while True:

    ret, img = cap.read()
    
    if not ret:
        
        break

    
    classIds, confs, bbox = net.detect(img, confThreshold = 0.5)

    if len(classIds)!=0:
        for classID, confidence, box in zip(classIds, confs, bbox):
        
            cv2.rectangle(img, box,color=(0,255,0), thickness=3)
            cv2.putText(img, str(classNames[classID-1]), (box[0]+7, box[1]+38), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2)
    
    print(classIds, confs, bbox)

    cv2.namedWindow("A.Rafay", cv2.WINDOW_NORMAL)
    cv2.imshow("A.Rafay", img)
    
    cv2.waitKey(1)



