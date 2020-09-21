#detection with facedetectionSSD
from FaceDetectionSSD import FaceDetectionSSD
from imutils import face_utils
import os, time
import cv2
import matplotlib.pyplot as plt
import dlib

 
shape_predictor = r"models\\shape_predictor_68_face_landmarks.dat"
# detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
detection = FaceDetectionSSD()
timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
OUT_DIR = f"images\\output\\{timestr}"
os.mkdir(OUT_DIR)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# cap.set(3, 1280)
# cap.set(4, 720)
cc = 0
while True:
    ret, frame = cap.read()
    if ret:
        rgb_img = frame
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        boxes = detection.detect_faces(rgb_img)

        for (x,y,w,h) in boxes:
    #         rect = [(x,y),(x+w, y+h)]
            # rgb_img = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2RGB)
            rect = dlib.rectangle(x,y,x+w,y+h)
            # print(rect)
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            points = [17, 26]
            # for i, (x, y) in enumerate(shape):
            #     if i in points:
            #         cv2.circle(rgb_img, (x, y), 5, (0, 0, 255), -1)
            #     else:
            #         cv2.circle(rgb_img, (x, y), 4, (0, 255, 0), -1)

            x1, x2 = shape[17][0], shape[26][0]
            y1 = shape[19][1]
            y3 = shape[46][1]
            y1cap, y3cap = y1 - int(h*0.08), y3 + int(h*0.1)
            x1cap, x2cap = x1-int(w*0.08), x2+int(w*0.08)
            cv2.rectangle(rgb_img, (x1cap,y1cap), (x2cap, y3cap), (255, 0, 255), 3)
            # cv2.circle(rgb_img, (x1, y1), 4, (255, 0, 255), -1)
            # cv2.circle(rgb_img, (x2, y2), 4, (255, 0, 255), -1)
            # (x1, y1), (x2, y2), (x3,y3) = shape[28], shape[29], shape[19]
            # cv2.circle(rgb_img, (x1, (y1+y2)//2), 4, (255, 0, 255), -1)
            # cv2.rectangle(rgb_img, (x-int(w*0.01),y3-int(h*0.08)), (x+w+int(w*0.01), (y1+y2)//2), (0,255,0), 3)
            # cv2.imwrite(OUT_DIR+f"\\{cc}.jpg", rgb_img[y1cap:y3cap, x1cap:x2cap])
            cc += 1
            
        # rgb_img = cv2.cvtColor(rgb_img,cv2.COLOR_RGB2BGR)
        cv2.imshow("s", rgb_img)    
        if cv2.waitKey(1) == 13:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()