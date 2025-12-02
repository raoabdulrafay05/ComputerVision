import mediapipe as mp
import cv2
import supervision as sv


class PoseDetection:
    def __init__(self):
        
        self.mpPose = mp.solutions.pose

        self.pose = self.mpPose.Pose()

        self.video_info = sv.VideoInfo.from_video_path("PoseDetection/vid4.mp4")

        self.draw = mp.solutions.drawing_utils

    def draw_pose(self, img):
        
        result = self.pose.process(img)
        if result.pose_landmarks:
            self.draw.draw_landmarks(
            img,
            result.pose_landmarks,
            self.mpPose.POSE_CONNECTIONS
        )
                
        return img


Pose = PoseDetection()

frames_generator = sv.get_video_frames_generator("PoseDetection/vid4.mp4")
with sv.VideoSink(target_path="PoseDetection/out4.mp4", video_info=Pose.video_info) as sink:
    
    for frame in frames_generator:
        
        new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        new_img = Pose.draw_pose(new_frame)
        
        final_frame = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("vid1" , final_frame)
        
        sink.write_frame(final_frame)
        
        cv2.waitKey(1)