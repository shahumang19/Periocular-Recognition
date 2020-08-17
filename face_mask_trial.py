#detection with facedetectionSSD
from FaceDetectionSSD import FaceDetectionSSD
from imutils import face_utils
import os
import cv2
import matplotlib.pyplot as plt
import dlib

 
shape_predictor = r"models\\shape_predictor_68_face_landmarks.dat"
# detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
detection = FaceDetectionSSD()
image_folder = r"images"

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# cap.set(3, 1280)
# cap.set(4, 720)

while True:
    ret, frame = cap.read()
    if ret:
        rgb_img = frame
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        boxes = detection.detect_faces(rgb_img)

        for (x,y,w,h) in boxes:
    #         rect = [(x,y),(x+w, y+h)]
            # rgb_img = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2RGB)
            cv2.rectangle(rgb_img, (x,y), (x+w, y+h), (0,255,0), 3)
            rect = dlib.rectangle(x,y,x+w,y+h)
            # print(rect)
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            for i, (x, y) in enumerate(shape):
                if i == 28 or i == 2 or i== 14:
                    cv2.circle(rgb_img, (x, y), 5, (0, 0, 255), -1)
                else:
                    cv2.circle(rgb_img, (x, y), 4, (0, 255, 0), -1)

        # rgb_img = cv2.cvtColor(rgb_img,cv2.COLOR_RGB2BGR)
        cv2.imshow("s", rgb_img)    
        if cv2.waitKey(1) == 13:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()