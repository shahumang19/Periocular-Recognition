from FaceEyeDetection import FaceEyeDetectionDlib
import cv2, os
import time
import numpy as np


BASE_PATH = r"images\ubipr\custom"
EYE_SAMPLES = 15

def save_left_right_eyes(BASE_DIR,data):
    for key, val in data.items():
        # print(key, val)
        cv2.imwrite(f"{BASE_DIR}\\le\\{key}", val[0])
        cv2.imwrite(f"{BASE_DIR}\\re\\{key}", val[1])


if __name__ == "__main__":
    count, fc = 0, 0
    eyes_dict = {}

    fd = FaceEyeDetectionDlib(model="models\\shape_predictor_68_face_landmarks.dat")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)


    WindowName="MyWindow"
    # cap = cv2.VideoCapture("data\\vid2.MP4")
    
    try:
        while True:
            ret, frame = cap.read()
            if ret:
                rects, face_boxes = fd.detect_faces(frame)
                all_boxes = fd.detect_eyes(frame, rects)

                reyew, reyeh = all_boxes[0][1][2:4]
                leyew, leyeh = all_boxes[0][2][2:4]

                extracted_eyes = fd.extract_faces_eye(frame, all_boxes)

                if len(all_boxes) > 0:
                    fc += 1
                    print(fc)

                for ix, (re, le) in enumerate(extracted_eyes):
                    timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
                    eyes_dict[f"{timestr}_{fc}.jpg"] = (le.copy(), re.copy())
                    # l_eye_list[f"{current_dir}\\le\\{timestr}_{fc}.jpg"] = le
                    # r_eye_list[f"{current_dir}\\re\\{timestr}_{fc}.jpg"] = re
                    # cv2.imshow(f"aligned{ix}", le)
                frame = fd.draw_faces_eyes(frame, all_boxes)

                if frame is None:
                    exit(1)

                cv2.imshow(WindowName, frame)

                # count += 15 # i.e. at 30 fps, this advances one second
                # cap.set(1, count)
                if cv2.waitKey(1) == 13 or fc >= EYE_SAMPLES:
                    break
            else:
                print("[INFO]  Video frame not available...")
                break
    except Exception as e:
        print(f"[ERROR] : {e}")
    finally:
        if len(eyes_dict) > 0:
            name = input("Enter name of the user : ")
            current_dir = f"{BASE_PATH}\\{name}"
            ix = 1
            while os.path.exists(current_dir):
                current_dir = f"{BASE_PATH}\\{name}({ix})"
                ix += 1
            os.mkdir(current_dir)
            os.mkdir(f"{current_dir}\\le")
            os.mkdir(f"{current_dir}\\re")
            save_left_right_eyes(current_dir, eyes_dict)
            print(f"Data Saved as --- {current_dir} ---")


        cap.release()
        cv2.destroyAllWindows()