import cv2
import supervision as sv
from ultralytics import YOLO

START = sv.Point(320, 0)
END = sv.Point(320, 480)

def main():

    model = YOLO("yolov8m.pt")

    line_zone = sv.LineZone(start=START, end=END)
    line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=1)

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=2)

    for result in model.track(source=0, stream=True):

        frame = result.orig_img

        detections = sv.Detections.from_ultralytics(result)

        # Safely extract tracker IDs
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        else:
            detections.tracker_id = [None] * len(detections)

        # Correct formatting for labels
        labels = [
            f"ID:{track_id if track_id is not None else '-'} "
            f"{model.names[class_id]} {confidence:.2f}"
            for class_id, confidence, track_id in zip(
                detections.class_id,
                detections.confidence,
                detections.tracker_id
            )
        ]

        annotated = box_annotator.annotate(scene=frame, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        # Line crossing
        line_zone.trigger(detections=detections)
        line_zone_annotator.annotate(frame=annotated, line_counter=line_zone)

        cv2.imshow("yolov8", annotated)

        if cv2.waitKey(30) == 27:
            break


if __name__ == "__main__":
    main()
