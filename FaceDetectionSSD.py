import cv2, numpy as np
from math import ceil


image_mean = np.array([127, 127, 127])
image_std = 128.0
iou_threshold = 0.3
center_variance = 0.1
size_variance = 0.2
min_boxes = [[10.0, 16.0, 24.0], [32.0, 48.0], [64.0, 96.0], [128.0, 192.0, 256.0]]
strides = [8.0, 16.0, 32.0, 64.0]


class FaceDetectionSSD():
    image_mean = np.array([127, 127, 127])
    image_std = 128.0
    iou_threshold = 0.3
    center_variance = 0.1
    size_variance = 0.2
    min_boxes = [[10.0, 16.0, 24.0], [32.0, 48.0], [64.0, 96.0], [128.0, 192.0, 256.0]]
    strides = [8.0, 16.0, 32.0, 64.0]
    
    def __init__(self, size = (320,240), onnx_path="models\\version-RFB-320_simplified.onnx"):
        """ Loads and returns model for feature face detection """
        try:
            self.net = cv2.dnn.readNetFromONNX(onnx_path)
            self.priors = self.define_img_size((320, 240))
            print("[INFO] Face Detection model loaded")
        except Exception as e:
            print(f"[ERROR] model {onnx_path} not found... : {e}")
            
   
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
        

    def detect_faces(self, image):
        """ Accepts single image and returns locations of detected faces """
        face_locations = []
        try:
            face_locations = self.inference(image)
        except Exception as e:
            print(f"[ERROR] FaceDetection detect_faces : {e}")
        return face_locations
        
        
    @staticmethod
    def extract_faces(image,boxes):
        """
        Accepts single image and detected face boxes and returns cropped face images.
        image : single image
        boxes : x,y,w,h of face detected i.e face coordinates
        """
        detected = []
        
        try:
            image_cp = image.copy()
            for (x,y,w,h) in boxes:
                cropped = image_cp[y:y+h, x:x+w]
                detected.append(cropped)
        except Exception as e:
            print(f"[ERROR] FaceDetection extract_faces : {e}")
            
        return detected
        
    @staticmethod
    def draw_faces(image, boxes):
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
    count = 0
    fd = FaceDetectionSSD()
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    try:
        while True:
            ret, frame = cap.read()
            if ret:
                boxes = fd.detect_faces(frame)
                frame = fd.draw_faces(frame, boxes)
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
    except Exception as e:
        print(f"[ ERROR ] : {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()    