import detect_faceMesh as faceMesh
import cv2
import supervision as sv


def main():
    video_info = sv.VideoInfo.from_video_path("FaceMesh/vid2.mp4")

    with sv.VideoSink(target_path="FaceMesh/out2.mp4", video_info=video_info) as sink:
        face_mesh_detector = faceMesh.DetectFaceMesh()
        
        for frame in sv.get_video_frames_generator("FaceMesh/vid2.mp4"):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame = face_mesh_detector.draw_mesh(rgb_frame)
            bgr_annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            sink.write_frame(bgr_annotated_frame)
            
if __name__ == "__main__":
    main()