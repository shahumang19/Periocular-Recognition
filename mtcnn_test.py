from mtcnn import MTCNN
import cv2, numpy as np
import sys

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model("face-mask-detector\\mask_detector.model")


def has_mask(image, maskNet):
    face = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    (mask, withoutMask) = maskNet.predict(face)[0]
    return True if mask > withoutMask else False

detector = MTCNN()

count = 0
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

try:
    while True:
        ret, frame = cap.read()

        if ret:
            raw = frame
            img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            
            results = detector.detect_faces(img)
            # print(results)

            for result in results:
                x,y,w,h = result["box"]
                # if w> 50 and h > 50 and img is not None:
                #     cv2.imshow("face", img[y:y+h, x:x+w])
                nose = list(result["keypoints"]["nose"])
                color = None
                if has_mask(img[y:y+h, x:x+w].copy(), maskNet):
                    nose[1] -= int(nose[1] * 0.1)
                    color = (0,255,0)
                else:
                    nose[1] -= int(nose[1] * 0.001)
                    color = (0,0,255)

                y += int(h*0.2)
                
                # print(nose)
                cv2.rectangle(raw, (x,y), (x+w, nose[1]), color, 4)
                # cv2.rectangle(raw, (x,y), ((x+w), nose[1]), (0,0,255), 10)

                for kp in result["keypoints"].values():
                    cv2.circle(raw, kp, 5, (255, 0, 255), -1)

            cv2.imshow("www", raw)

            # count += 15
            # cap.set(0, count)

            if cv2.waitKey(1) == 13:
                break
        else:
            break
except Exception as e:
    print(f"[Error] Line - {sys.exc_info()[-1].tb_lineno}: {e} ")
finally:
    cap.release()
    cv2.destroyAllWindows()