import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
        static_image_mode=self.mode,
        max_num_hands=self.maxHands,
        min_detection_confidence=self.detectionCon,
        min_tracking_confidence=self.trackCon
        )

        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        handType = "NoHand"
        left = False
        right = False
        if self.results.multi_hand_landmarks:
            for i, handLms in enumerate(self.results.multi_hand_landmarks):
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                handType = self.results.multi_handedness[i].classification[0].label
                if str(handType) == "Right":
                    right = True
                if str(handType) == "Left":
                    left = True
        return img, right, left

    def findPosition(self, img, idx=0, draw=True):

        self.lmList = []
        if self.results.multi_hand_landmarks:
            for idx,myHand in enumerate(self.results.multi_hand_landmarks):
                # print(idx)
                myHand = self.results.multi_hand_landmarks[idx]
                for id, lm in enumerate(myHand.landmark):
                    # print(id, lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id, cx, cy)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return self.lmList
    
    
    def findTwoFingerPosition(self, idx1, idx2, img, draw=True):
        
        x,y = 0,0
        landmark = None
        if self.lmList:
            x1, y1 = self.lmList[idx1][1], self.lmList[idx1][2]
            x2, y2 = self.lmList[idx2][1], self.lmList[idx2][2]
            x = (x1 + x2) // 2
            y = (y1 + y2) // 2
            
            if draw:
                cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.circle(img, (x, y), 10, (0, 0, 255), cv2.FILLED)
                
        if landmark is None:
            return None, None
        return x, y
                    


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        frame = img.copy()
        frame, right, left = detector.findHands(img)
        lmList = detector.findPosition(img)
        x_val , y_val = detector.findTwoFingerPosition(8,12,img)
        count = 10
        
        if not right :
            count = count - 5
            
        if not left:
            count = count - 5
        
        if len(lmList) != 0:
            if left:
                if lmList[5] and lmList[17] and lmList[4]:
                    if lmList[5][1] >= lmList[4][1] <= (lmList[5][1] + lmList[17][1]):
                        # print("Thumb is closed") 
                        count-=1
                        
                    if lmList[5][2] <= lmList[8][2] <= lmList[0][2]:
                        # print("Right finger is closed")
                        count-=1
                    
                    if lmList[9][2] <= lmList[12][2] <= lmList[0][2]:
                        # print("Middle finger is closed") 
                        count-=1
                        
                    if lmList[13][2] <= lmList[16][2] <= lmList[0][2]:
                        # print("Ring finger is closed") 
                        count-=1
                    
                    if lmList[17][2] <= lmList[20][2] <= lmList[0][2]:
                        # print("Tiny finger is closed")
                        count-=1 
                print(count)
                

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 0), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()