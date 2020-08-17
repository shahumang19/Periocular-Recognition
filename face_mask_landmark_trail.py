#detection with facedetectionSSD
from FaceDetectionSSD import FaceDetectionSSD
from imutils import face_utils
import os
import cv2
import numpy as np


LBFmodel = "models\\lbfmodel.yaml"

landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)

# shape_predictor = r"models\\shape_predictor_68_face_landmarks.dat"
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(shape_predictor)
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
        image_gray = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2GRAY)
        if len(boxes) > 0:
            _, landmarks = landmark_detector.fit(image_gray, np.array(boxes))

            for (x,y,w,h) in boxes:
                # rect = [(x,y),(x+w, y+h)]
                # rgb_img = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2RGB)
                cv2.rectangle(rgb_img, (x,y), (x+w, y+h), (0,255,0), 3)

            for landmark in landmarks:
                for x,y in landmark[0]:
                    cv2.circle(rgb_img, (x, y), 1, (255, 255, 255), 1)

        # rgb_img = cv2.cvtColor(rgb_img,cv2.COLOR_RGB2BGR)
        cv2.imshow("s", rgb_img)    
        if cv2.waitKey(1) == 13:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()