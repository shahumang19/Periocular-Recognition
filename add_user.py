from FaceDetectionDlib import FaceDetectAndAlign
from imutils import resize
import cv2, os
import time
import numpy as np


BASE_PATH = r"images\custom"
FACE_SAMPLES = 15


def show_text_center(img, text):
    img1 = img.copy()
    h, w, _ = img1.shape
    cv2.putText(img1, text, (h//2 + 100 , w//2), 0, 20, (0, 0, 255), 5, cv2.LINE_AA)
    return img1


def save_faces(BASE_DIR,data):
    for key, val in data.items():
        cv2.imwrite(f"{BASE_DIR}\\{key}", val)


if __name__ == "__main__":
    count, fc, max_wait_count = 0, 0, 5
    face_dict = {}

    fd =  FaceDetectAndAlign(desiredFaceWidth=256)
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)


    WindowName="MyWindow"
    # cap = cv2.VideoCapture("data\\vid2.MP4")
    
    try:
        start = time.time()
        while True:
            ret, frame = cap.read()
            if ret:
                if max_wait_count > 0:
                    # print(time.time())
                    if int(time.time() - start) == 1:
                        max_wait_count -= 1
                        start = time.time()

                    frame = show_text_center(frame, str(max_wait_count))
                    # time.sleep(1)
                    # print(wait_count)
                else:
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