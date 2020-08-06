from FaceDetectionDlib import FaceDetectAndAlign
import cv2
import time
import numpy as np



if __name__ == "__main__":
    count = 0
    fd = FaceDetectAndAlign()
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    # cap = cv2.VideoCapture("data\\vid2.MP4")
    
    try:
        while True:
            ret, frame = cap.read()
            if ret:
                boxes = fd.detect_faces(frame)
                aligned_faces = fd.extract_faces(frame, boxes)
                for ix, face in enumerate(aligned_faces):
                    h, w, _ = face.shape
                    starth = int(h*0.05)
                    endh = h - int(h*0.58)
                    face = face[starth:endh]
                    timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
                    cv2.imwrite(f"images\\{timestr}.jpg", face)
                    cv2.imshow(f"aligned{ix}", face)
                frame = fd.draw_faces(frame, boxes)
                if frame is None:
                    exit(1)
                cv2.imshow("Live Face Detection", frame)
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