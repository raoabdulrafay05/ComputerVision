from ultralytics import YOLO
import cv2
import supervision as sv

model = YOLO("yolov8m.pt")

cap = cv2.VideoCapture(0)

cap.set(0, 480)
cap.set(0, 320)

while True:
    
    ret, img = cap.read()
    
    result = model.predict(img)[0]

    detections = sv.Detections.from_ultralytics(result)
    
    box_annotator = sv.BoxAnnotator(thickness=2)
    
    label_annotator = sv.LabelAnnotator(text_scale=2)
    
    new_img = box_annotator.annotate(scene=img, detections=detections)
    
    labels = [f"{confidence:.2f},{model.model.names[class_id]}"  for class_id, confidence in zip(detections.class_id, detections.confidence)]
    
    new_img = label_annotator.annotate(scene=new_img, detections=detections, labels=labels)
    
    
    
    cv2.imshow("image", new_img)
    
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
    
cap.release()
cv2.destroyAllWindows()