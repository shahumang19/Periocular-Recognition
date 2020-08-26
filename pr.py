from FaceDetectionDlib import FaceDetectAndAlign
from Facenet import Facenet
from imutils import resize
import numpy as np
import utils as u
import pickle
import cv2


if __name__ == "__main__":
    count, fc = 0, 0

    fd =  FaceDetectAndAlign(align=True ,desiredFaceWidth=512)
    facenet = Facenet("models\\facenet_pr_keras.h5")

    data = u.read_data("data\\merged_face_features.pkl")
    labels = data["labels"]
    index = u.generate_annoyIndex_live(data["features"], trees=20)

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
                faces = fd.extract_faces(frame.copy(), rects)
                features = facenet.get_embeddings(faces)

                if features is not None:
                    predictions = u.get_predictions(features, index, labels, thresh=0.5, name_count_thresh=1)
                    print(predictions)
                    frame = u.draw_predictions(frame, fd.rect_to_bb(rects), predictions)

                if frame is None:
                    frame = np.zeros((640, 480))
                
                frame = resize(frame, width=640)
                cv2.imshow(WindowName, frame)

                # count += 15 # i.e. at 30 fps, this advances one second
                # cap.set(1, count)
                if cv2.waitKey(1) == 13:
                    break
            else:
                print("[INFO]  Video frame not available...")
                break
    except Exception as e:
        print(f"[ERROR] : {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()