from FaceDetectionDlib import FaceDetectAndAlign
from imutils import resize
import cv2, os
import time
import numpy as np


BASE_PATH = r"images\ubipr\custom"
FACE_SAMPLES = 15

def save_faces(BASE_DIR,data):
    for key, val in data.items():
        cv2.imwrite(f"{BASE_DIR}\\{key}", val)


if __name__ == "__main__":
    count, fc = 0, 0
    face_dict = {}

    fd =  FaceDetectAndAlign(align=False)
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)


    WindowName="MyWindow"
    # cap = cv2.VideoCapture("data\\vid2.MP4")
    
    try:
        while True:
            ret, frame = cap.read()
            if ret:
                rects = fd.detect_faces(frame, biggest=True)
                faces = fd.extract_faces(frame, rects)

                if len(faces) > 0:
                    fc += 1
                    print(fc)

                for face in faces:
                    timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
                    face_dict[f"{timestr}_{fc}.jpg"] = face.copy()
                    
                frame = fd.draw_faces(frame, rects)

                if frame is None:
                    exit(1)
                
                frame = resize(frame, width=640)
                cv2.imshow(WindowName, frame)

                # count += 15 # i.e. at 30 fps, this advances one second
                # cap.set(1, count)
                if cv2.waitKey(1) == 13 or fc >= FACE_SAMPLES:
                    break
            else:
                print("[INFO]  Video frame not available...")
                break
    except Exception as e:
        print(f"[ERROR] : {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        if len(face_dict) > 0:
            name = input("Enter name of the user : ")
            current_dir = f"{BASE_PATH}\\{name}"
            ix = 1

            while os.path.exists(current_dir):
                current_dir = f"{BASE_PATH}\\{name}({ix})"
                ix += 1

            os.mkdir(current_dir)
            save_faces(current_dir, face_dict)
            print(f"Data Saved as --- {current_dir} ---")