from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2


class FaceDetectAndAlign:
    def __init__(self, shape_predictor_path="models\\shape_predictor_68_face_landmarks.dat", align=True):
        try:
            self.detector = dlib.get_frontal_face_detector()
            if align:
                self.predictor = dlib.shape_predictor(shape_predictor_path)
                self.aligner = FaceAligner(self.predictor, desiredFaceWidth=400)
            self.align = align
        except Exception as e:
            print(f"[ERROR] {self.__class__.__name__} model {shape_predictor_path} not found... : {e}")

    def detect_faces(self, image, biggest=True):
        rects = []
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)
            if len(rects) > 0:
                rects = sorted(rects, key=lambda rect: rect_to_bb(rect)[2]*rect_to_bb(rect)[3], reverse=True)[0:1]
        except Exception as e:
            print(f"[ERROR] {self.__class__.__name__} detect_faces : {e}")
        return rects

    def extract_faces(self, image, rects):
        extracted_faces = []
        try:
            if self.align:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                for rect in rects:
                    faceAligned = self.aligner.align(image, gray, rect)
                    faceAlignedGray = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)
                    rects1 = self.detector(faceAlignedGray, 2)
                    if len(rects1) > 0:
                        (x1, y1, w1, h1) = rect_to_bb(rects1[0])
                        faceAligned = faceAligned[y1:y1 + h1//2, x1:x1 + w1]
                    extracted_faces.append(faceAligned)
            else:
                for rect in rects:
                    (x1, y1, w1, h1) = rect_to_bb(rect)
                    face = image[y1:y1 + h1//2, x1:x1 + w1].copy()
                    extracted_faces.append(face)
        except Exception as e:
            print(f"[ERROR] {self.__class__.__name__} extract_faces : {e}")

        return extracted_faces

    @classmethod
    def rect_to_bb(cls, rects):
        bbs = []
        try:
            bbs = [rect_to_bb(rect) for rect in rects]
        except Exception as e:
            print(f"[ERROR] {cls.__name__} rect_to_bb : {e}")
        return bbs

    @classmethod
    def draw_faces(cls, image, rects):
        image_cp = None
        try:
            image_cp = image.copy()
            for rect in rects:
                (x, y, w, h) = rect_to_bb(rect)
                cv2.rectangle(image_cp, (x,y), (x+w, y+h), (0,0,255), int(0.01*image_cp.shape[0]))
        except Exception as e:
            print(f"[ERROR] {cls.__name__} {e}")
            
        return image_cp


if __name__ == "__main__":
    count = 0
    fd = FaceDetectAndAlign()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap = cv2.VideoCapture("data\\vid2.MP4")
    
    try:
        while True:
            ret, frame = cap.read()
            if ret:
                boxes = fd.detect_faces(frame)
                aligned_faces = fd.extract_faces(frame, boxes)
                for ix, face in enumerate(aligned_faces):
                    # h, w, _ = face.shape
                    # face = face[0:h//2]
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