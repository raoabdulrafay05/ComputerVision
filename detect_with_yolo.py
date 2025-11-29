from ultralytics import YOLO
import supervision as sv
import cv2

model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture("vid.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("vid_annotated.mp4", fourcc, fps, (width, height))

box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=1, text_thickness=2)

while True: 
    
    ret, frame = cap.read()
    if not ret:
        break
    
    result = model(frame, conf = 0.25)[0]
    
    detections = sv.Detections.from_ultralytics(result)
    
    print(detections)
    
    labels = [f"{model.names[class_id]} {conf:.2f}" for class_id , conf in zip(detections.class_id, detections.confidence)]
    
    annotated = box_annotator.annotate(scene=frame, detections=detections)
    
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
    
    out.write(annotated)
    

cap.release()
out.release()

print("DONE")