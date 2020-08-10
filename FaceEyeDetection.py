from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2


class FaceEyeDetectionDlib:
    def __init__(self, model="shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model)
        print("[INFO] Face Landmark Detection Model loaded....")

        self.left_eye_index = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self.right_eye_index = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    @staticmethod
    def _process_eye_box_(box):
        box = list(box)
        box[1] -= int(box[3])
        box[3] += int(box[3])*2

        box[0] -= int(box[2]*0.2)
        box[2] += int(box[2]*0.2)*2
        return box

    @staticmethod
    def _preprocess_face_box_(rect):
        box = face_utils.rect_to_bb(rect)
        box = list(box)
        box[1] -= int(box[3]*0.1)
        box[3] += int(box[3]*0.1)
        return box

    def detect_faces(self, image):
        rects = []
        if image is not None:
            gray = image
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale frame
            rects = self.detector(gray, 0)
            bbs = [FaceEyeDetectionDlib._preprocess_face_box_(rect) for rect in rects]
        return rects, bbs

    def detect_eyes(self, image, face_rects):
        face_landmarks = []
        if image is not None:
            gray = image
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for rect in face_rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            left_eye_box = cv2.boundingRect(np.array([shape[self.left_eye_index[0]:self.left_eye_index[1]]]))
            left_eye_box = FaceEyeDetectionDlib._process_eye_box_(left_eye_box)
            right_eye_box = cv2.boundingRect(np.array([shape[self.right_eye_index[0]:self.right_eye_index[1]]]))
            right_eye_box = FaceEyeDetectionDlib._process_eye_box_(right_eye_box)

            face_landmarks.append((FaceEyeDetectionDlib._preprocess_face_box_(rect),left_eye_box, right_eye_box))
        return face_landmarks
    
    def extract_faces(self, image, shapes):
        images = []
        for face in shapes:
            x,y,w,h = face
            face = image[y:y+h, x:x+w]
            images.append(face)
        return images

    def extract_faces_eye(self, image, shapes, extract_face=False):
        images = []
        for faceBB, leyeBB, reyeBB in shapes:
            if extract_face:
                x,y,w,h = faceBB
                face = image[y:y+h, x:x+w]
                x,y,w,h = leyeBB
                leye = image[y:y+h, x:x+w]
                x,y,w,h = reyeBB
                reye = image[y:y+h, x:x+w]
                images.append((face, leye, reye))
            else:
                x,y,w,h = leyeBB
                leye = image[y:y+h, x:x+w]
                x,y,w,h = reyeBB
                reye = image[y:y+h, x:x+w]
                images.append((leye, reye))
        return images

    def draw_faces_eyes(self, image, boxes):
            """
            Accepts single image and detected face boxes and returns image with rectangles drawn on face locations.
            image : single image
            boxes : x,y,w,h of face detected i.e face coordinates
            """
            image_cp = None
            try:
                image_cp = image.copy()
                # print(boxes)
                for (faceBB, leyeBB, reyeBB) in boxes:
                    # print(1)
                    (x,y,w,h) = faceBB
                    cv2.rectangle(image_cp, (x,y), (x+w, y+h), (0,0,255), int(0.01*image_cp.shape[0]))
                    (x,y,w,h) = leyeBB
                    cv2.rectangle(image_cp, (x,y), (x+w, y+h), (0,0,255), int(0.01*image_cp.shape[0]))
                    (x,y,w,h) = reyeBB
                    cv2.rectangle(image_cp, (x,y), (x+w, y+h), (0,0,255), int(0.01*image_cp.shape[0]))

            except Exception as e:
                print(f"[ERROR] {e}")
                
            return image_cp


if __name__ == "__main__":
    from os.path import join

    count = 0
    fd = FaceEyeDetectionDlib(join("models","shape_predictor_68_face_landmarks.dat"))
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    try:
        while True:
            ret, frame = cap.read()
            if ret:
                rects, boxes = fd.detect_faces(frame)
                # faces = fd.extract_faces(frame, boxes)
                boxes = fd.detect_eyes(frame, rects)
                # fld = fd.extract_faces_eye(frame, boxes, True)
                # for ix, face in enumerate(faces):
                #     cv2.imshow(f"face{ix}", face)

                # for ix, (face, le, re) in enumerate(fld):
                #     cv2.imshow(f"facex{ix}", face)
                #     cv2.imshow(f"left{ix}", le)
                #     cv2.imshow(f"right{ix}", re)

                frame = fd.draw_faces_eyes(frame, boxes)
                if frame is None:
                    exit(1)
                cv2.imshow("Live Face Detection", frame)
                # count += 15 # i.e. at 30 fps, this advances one second
                # cap.set(1, count)
                if cv2.waitKey(1) == 13:
                    break
            else:
                print("[ INFO ]  Video frame not available...")
                break
    except Exception as e:
        print(f"[ ERROR ] : {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()    

        
    