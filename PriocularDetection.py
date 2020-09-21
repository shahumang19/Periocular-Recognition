#Have use ssd and dlib for detecting periocular region

from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb, shape_to_np
from math import ceil
import imutils
import dlib
import cv2, numpy as np

image_mean = np.array([127, 127, 127])
image_std = 128.0
iou_threshold = 0.3
center_variance = 0.1
size_variance = 0.2
min_boxes = [[10.0, 16.0, 24.0], [32.0, 48.0], [64.0, 96.0], [128.0, 192.0, 256.0]]
strides = [8.0, 16.0, 32.0, 64.0]


class PeriocularDetection():
    
    def __init__(self, size = (320,240), 
                        onnx_path="models\\version-RFB-320_simplified.onnx",
                        shape_predictor_path="models\\shape_predictor_68_face_landmarks.dat", 
                        align=True, desiredFaceWidth=400):
        """ Loads and returns model for feature face detection """
        try:
            self.net = cv2.dnn.readNetFromONNX(onnx_path)
            self.priors = self.define_img_size((320, 240))
            self.align = align
            if align:
                self.predictor = dlib.shape_predictor(shape_predictor_path)
                self.aligner = FaceAligner(self.predictor, desiredFaceWidth=desiredFaceWidth)

            print("[INFO] Face Detection model loaded")
        except Exception as e:
            print(f"[ERROR] model {onnx_path} or {shape_predictor_path} not found... : {e}")
            
   
    def define_img_size(self,image_size):
        shrinkage_list = []
        feature_map_w_h_list = []
        for size in image_size:
            feature_map = [int(ceil(size / stride)) for stride in strides]
            feature_map_w_h_list.append(feature_map)

        for i in range(0, len(image_size)):
            shrinkage_list.append(strides)
        priors = self.generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes)
        return priors


    def generate_priors(self, feature_map_list, shrinkage_list, image_size, min_boxes):
        priors = []
        for index in range(0, len(feature_map_list[0])):
            scale_w = image_size[0] / shrinkage_list[0][index]
            scale_h = image_size[1] / shrinkage_list[1][index]
            for j in range(0, feature_map_list[1][index]):
                for i in range(0, feature_map_list[0][index]):
                    x_center = (i + 0.5) / scale_w
                    y_center = (j + 0.5) / scale_h

                    for min_box in min_boxes[index]:
                        w = min_box / image_size[0]
                        h = min_box / image_size[1]
                        priors.append([
                            x_center,
                            y_center,
                            w,
                            h
                        ])
                        
        return np.clip(priors, 0.0, 1.0)


    def hard_nms(self, box_scores, iou_threshold, top_k=-1, candidate_size=200):
        scores = box_scores[:, -1]
        boxes = box_scores[:, :-1]
        picked = []
        indexes = np.argsort(scores)
        indexes = indexes[-candidate_size:]
        while len(indexes) > 0:
            current = indexes[-1]
            picked.append(current)
            if 0 < top_k == len(picked) or len(indexes) == 1:
                break
            current_box = boxes[current, :]
            indexes = indexes[:-1]
            rest_boxes = boxes[indexes, :]
            iou = self.iou_of(
                rest_boxes,
                np.expand_dims(current_box, axis=0),
            )
            indexes = indexes[iou <= iou_threshold]
        return box_scores[picked, :]


    def area_of(self, left_top, right_bottom):
        hw = np.clip(right_bottom - left_top, 0.0, None)
        return hw[..., 0] * hw[..., 1]

    

    def iou_of(self, boxes0, boxes1, eps=1e-5):
        overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

        overlap_area = self.area_of(overlap_left_top, overlap_right_bottom)
        area0 = self.area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = self.area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)

    
    def predict(self, width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = self.hard_nms(box_probs,
                                 iou_threshold=iou_threshold,
                                 top_k=top_k,
                                 )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

    
    def convert_locations_to_boxes(self, locations, priors, center_variance,
                                   size_variance):
        if len(priors.shape) + 1 == len(locations.shape):
            priors = np.expand_dims(priors, 0)
        return np.concatenate([
            locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
            np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
        ], axis=len(locations.shape) - 1)

    
    def center_form_to_corner_form(self, locations):
        return np.concatenate([locations[..., :2] - locations[..., 2:] / 2,
                               locations[..., :2] + locations[..., 2:] / 2], len(locations.shape) - 1)

    
    def inference(self, image):
        input_size = (320, 240)
        witdh = input_size[0]
        height = input_size[1]
        locations = []

        rect = cv2.resize(image, (witdh, height))
        rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
        self.net.setInput(cv2.dnn.blobFromImage(rect, 1 / image_std, (witdh, height), 127))
        boxes, scores = self.net.forward(["boxes", "scores"])

        boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
        scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
        boxes = self.convert_locations_to_boxes(boxes, self.priors, center_variance, size_variance)
        boxes = self.center_form_to_corner_form(boxes)
        boxes, labels, probs = self.predict(image.shape[1], image.shape[0], scores, boxes, 0.7)
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            w = box[2]-box[0]
            h = box[3]-box[1]
            # box[0] -= int(w*0.15)
            #box[1] -= int(h*0.15)
            # w += int(w*0.15)*2
            #h += int(h*0.15)*2
            locations.append((box[0], box[1], w, h))
        return locations
        
    def bb_to_rect(self, bbs):
        rects = []
        if bbs is not None:
            for (x, y, w, h) in bbs:
                rects.append(dlib.rectangle(x,y,x+w,y+h))
        return rects


    def detect_faces(self, image, biggest=False):
        """ Accepts single image and returns locations of detected faces """
        face_locations = []
        try:
            face_locations = self.inference(image)
            if len(face_locations) > 0:
                face_locations = sorted(face_locations, key=lambda bb: bb[2]*bb[3], reverse=True)[0:1]
            # face_locations = self.bb_to_rect(face_locations)
        except Exception as e:
            print(f"[ERROR] FaceDetection detect_faces : {e}")
        return face_locations

    def get_face_pr_boxes(self, image, biggest=False):
        pr_locations, face_locations = [], []
        face_locations = self.detect_faces(image, biggest)
        rects = self.bb_to_rect(face_locations)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for rect, (x, y, w, h) in zip(rects, face_locations):
            print(rect, gray.shape)
            pr_locations.append(self._predict_pr_region(gray[y:y+h, x:x+w].copy(), rect))
        return face_locations, pr_locations

        
    def extract_faces(self, image, bboxes):
        """
        Accepts single image and detected face boxes and returns cropped face images.
        image : single image
        rects : dlib face rectangles
        """
        extracted_faces = []
        try:
            if self.align:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rects = self.bb_to_rect(bboxes)
                for rect in rects:
                    faceAligned = self.aligner.align(image, gray, rect) 
                    aligned_bboxes = self.detect_faces(faceAligned, True)
                    if len(aligned_bboxes) > 0:
                        (x1, y1, w1, h1) = aligned_bboxes[0]
                        faceAligned = faceAligned[y1:y1 + h1//2, x1:x1 + w1]
                    extracted_faces.append(faceAligned)
            else:
                for (x1, y1, w1, h1) in bboxes:
                    face = image[y1:y1 + h1//2, x1:x1 + w1].copy()
                    extracted_faces.append(face)
        except Exception as e:
            print(f"[ERROR] {self.__class__.__name__} extract_faces : {e}")
        finally:
            return extracted_faces

    def _predict_pr_region(self, grayFace, rect):
        w, h = rect_to_bb(rect)[2:4]
        shape = self.predictor(grayFace, [rect])
        shape = shape_to_np(shape)
        x1, x2 = shape[17][0], shape[26][0]
        y1, y2 = shape[19][1], shape[46][1]
        x1cap, x2cap = x1-int(w*0.08), x2+int(w*0.08)
        y1cap, y2cap = y1 - int(h*0.08), y2 + int(h*0.1)
        return x1cap, y1cap, (x2cap - x1cap), (y2cap - y1cap)

    def extract_periocular_region(self, image, bboxes):
        extracted_pr = []
        try:
            rects = self.bb_to_rect(bboxes)
            if self.align:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                for rect, (x, y, w, h) in zip(rects, bboxes):
                    faceAligned = self.aligner.align(image, gray, rect)
                    faceAlignedGray = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)
                    aligned_bboxes = self.detect_faces(faceAligned, True)

                    if len(aligned_bboxes) > 0:
                        rects1 = self.bb_to_rect(aligned_bboxes)
                        (x1, y1, w1, h1) = self._predict_pr_region(faceAlignedGray, rects1[0])
                    else:
                        (x1, y1, w1, h1) = self._predict_pr_region(faceAlignedGray, rect)

                    faceAligned = faceAligned[y1:y1 + h1//2, x1:x1 + w1]
                    extracted_pr.append(faceAligned)
            else:
                gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                for rect in rects:
                    (x1, y1, w1, h1) = self._predict_pr_region(gray_img, rect)
                    face = image[y1:y1 + h1//2, x1:x1 + w1].copy()
                    extracted_pr.append(face)
        except Exception as e:
            print(f"[ERROR] {self.__class__.__name__} extract_faces : {e}")
        finally:
            return extracted_pr

        
    @staticmethod
    def draw_region(image, boxes):
        """
        Accepts single image and detected face boxes and returns image with rectangles drawn on face locations.
        image : single image
        boxes : x,y,w,h of face detected i.e face coordinates
        """
        image_cp = None
        try:
            image_cp = image.copy()
            for (x,y,w,h) in boxes:
                cv2.rectangle(image_cp, (x,y), (x+w, y+h), (0,0,255), int(0.01*image_cp.shape[0]))
        except Exception as e:
            print(f"[ERROR] {e}")
            
        return image_cp
    
    
if __name__ == "__main__":
    import sys
    count = 0
    pd = PeriocularDetection()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # try:
    while True:
        ret, frame = cap.read()
        if ret:
            fbs, pbs = pd.get_face_pr_boxes(frame)
            frame = pd.draw_region(frame, pbs)
            if frame is None:
                exit(1)
            cv2.imshow("Live Face Detection", frame)
            count += 15 # i.e. at 30 fps, this advances one second
            cap.set(1, count)
            if cv2.waitKey(1) == 13:
                break
        else:
            print("[ INFO ]  Video frame not available...")
            break
    # except Exception as e:
    #     print(f"[ ERROR ] Line-{sys.exc_info()[-1].tb_lineno} : {e}")
    # finally:
    cap.release()
    cv2.destroyAllWindows()    