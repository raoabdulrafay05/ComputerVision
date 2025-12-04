import cv2
import mediapipe as mp

class DetectFaceMesh:
    
    def __init__(self):
        
        self.face = mp.solutions.face_mesh
        self.face_mesh = self.face.FaceMesh()
        self.draw = mp.solutions.drawing_utils
        self.draw_spec = self.draw.DrawingSpec(thickness=1, circle_radius=1,color = (0,255,0))
        
    def draw_mesh(self, img):
        
        self.result = self.face_mesh.process(img)
        
        if self.result.multi_face_landmarks:
            for fmask in self.result.multi_face_landmarks:
                
                self.draw.draw_landmarks(img, fmask, self.face.FACEMESH_CONTOURS,self.draw_spec, self.draw_spec)
        
        
        return img

    def get_landmarks(self, img):
        
        landmark_list = [[]]
        
        if self.result.multi_face_landmarks:
            for idx, fmask in enumerate(self.result.multi_face_landmarks):
                for id, lm in enumerate(fmask.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list[idx].append([id, cx, cy])
        return landmark_list
    
def main():
    
    cap = cv2.VideoCapture(0)
    detector = DetectFaceMesh()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        new_frame = detector.draw_mesh(new_frame)
        lm = detector.get_landmarks(new_frame)
        bgr_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR)
        if len(lm)!=0:
            print(lm)

        cv2.imshow("Face Mesh", bgr_frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()