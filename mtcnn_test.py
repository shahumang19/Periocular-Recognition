from mtcnn import MTCNN
import cv2

detector = MTCNN()

count = 0
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()

    if ret:
        raw = frame
        img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        
        results = detector.detect_faces(img)
        print(results)

        for result in results:
            x,y,w,h = result["box"]
            cv2.rectangle(raw, (x,y), (x+w, y+h), (0,0,255), 10)

            for kp in result["keypoints"].values():
                cv2.circle(raw, kp, 5, (255, 0, 255), -1)

        cv2.imshow("www", raw)

        count += 15
        cap.set(0, count)

        if cv2.waitKey(1) == 13:
            break


    else:
        break

cap.release()
cv2.destroyAllWindows()