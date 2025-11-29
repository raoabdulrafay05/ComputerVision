# from ultralytics import YOLO
# import supervision as sv
# import numpy as np
# from supervision.geometry.core import Point

# video_path = "car_tracking.mp4"
# model = YOLO("yolov8n.pt")

# POLYGON = np.array([
#     [759, 1023],
#     [318, 362],
#     [859, 341],
#     [1895, 815]
# ])

# LINE = np.array([[581, 633], [1548, 599]])

# video_info = sv.VideoInfo.from_video_path(video_path)
# box_annotator = sv.BoxAnnotator()
# label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=1)
# zone = sv.PolygonZone(polygon=POLYGON)
# tracker = sv.ByteTrack(frame_rate=video_info.fps)
# line_zone = sv.LineZone(
#     start=Point(LINE[0][0], LINE[0][1]),
#     end=Point(LINE[1][0], LINE[1][1]),
#     triggering_anchors=[sv.Position.BOTTOM_CENTER]
# )

# line_zone_annotator = sv.LineZoneAnnotator()

# with sv.VideoSink(target_path="display.mp4", video_info=video_info) as sink:
#     for frame in sv.get_video_frames_generator(video_path):
#         result = model(frame, conf=0.5)[0]
#         detections = sv.Detections.from_ultralytics(result)
#         tracked_detections = tracker.update_with_detections(detections=detections)

#         # Filter detections inside the polygon
#         inside_mask = zone.trigger(detections=tracked_detections)

#         detections_inside = tracked_detections[inside_mask]           
#         crossed_ids = line_zone.trigger(detections=tracked_detections)

                
#         # Annotate
#         new_frame = box_annotator.annotate(scene=frame.copy(), detections=detections_inside)
#         labels = [
#             f"ID: {tid} {model.names[cid]} {conf:.2f}"
#             for cid, conf, tid in zip(
#                 detections_inside.class_id,
#                 detections_inside.confidence,
#                 detections_inside.tracker_id
#             )
#         ]
#         new_frame = label_annotator.annotate(scene=new_frame, detections=detections_inside, labels=labels)
#         new_frame = line_zone_annotator.annotate(frame=new_frame, line_counter=line_zone)

        
#         sink.write_frame(new_frame)

from ultralytics import YOLO
import supervision as sv
import numpy as np
from supervision.geometry.core import Point
import cv2

video_path = "car_tracking.mp4"
model = YOLO("yolov8n.pt")

POLYGON = np.array([
    [759, 1023],
    [318, 362],
    [859, 341],
    [1895, 815]
])

LINE = np.array([[581, 633], [1548, 599]])

video_info = sv.VideoInfo.from_video_path(video_path)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=1)
zone = sv.PolygonZone(polygon=POLYGON)
tracker = sv.ByteTrack(frame_rate=video_info.fps)
line_zone = sv.LineZone(
    start=Point(LINE[0][0], LINE[0][1]),
    end=Point(LINE[1][0], LINE[1][1]),
    triggering_anchors=[sv.Position.BOTTOM_CENTER]
)

# Sets to track unique IDs
crossed_in_ids = set()
crossed_out_ids = set()

with sv.VideoSink(target_path="display.mp4", video_info=video_info) as sink:
    for frame in sv.get_video_frames_generator(video_path):
        result = model(frame, conf=0.5)[0]
        detections = sv.Detections.from_ultralytics(result)
        tracked_detections = tracker.update_with_detections(detections=detections)

        # Filter detections inside polygon
        inside_mask = zone.trigger(detections=tracked_detections)
        detections_inside = tracked_detections[inside_mask]

        # Trigger line zone
        crossed_in, crossed_out = line_zone.trigger(detections=tracked_detections)

        # Update sets with new IDs
        crossed_in_ids.update(tracked_detections.tracker_id[crossed_in])
        crossed_out_ids.update(tracked_detections.tracker_id[crossed_out])

        # Annotate boxes and labels
        new_frame = box_annotator.annotate(scene=frame.copy(), detections=detections_inside)
        labels = [
            f"ID: {tid} {model.names[cid]} {conf:.2f}"
            for cid, conf, tid in zip(
                detections_inside.class_id,
                detections_inside.confidence,
                detections_inside.tracker_id
            )
        ]
        new_frame = label_annotator.annotate(scene=new_frame, detections=detections_inside, labels=labels)

        # Draw counting line
        cv2.line(new_frame, tuple(LINE[0]), tuple(LINE[1]), (0, 0, 255), 2)

        # Display counts
        cv2.putText(
            new_frame,
            f"IN: {len(crossed_in_ids)}  OUT: {len(crossed_out_ids)}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3,
            cv2.LINE_AA
        )

        sink.write_frame(new_frame)
