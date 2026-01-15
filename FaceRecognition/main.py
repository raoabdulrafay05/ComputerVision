from deepface import DeepFace
import cv2

img = cv2.imread("FaceRecognition/img.jpg")
# Change "FaceNet512" to "Facenet512"
embeddings = DeepFace.represent(
    img_path="FaceRecognition/im.jpg", 
    model_name="Facenet512"
)

result = DeepFace.verify(img1_path = "FaceRecognition/im.jpg", img2_path = "FaceRecognition/image.png",model_name="Facenet512")


# print(embeddings)
print(result)