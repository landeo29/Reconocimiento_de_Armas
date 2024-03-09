import cv2
import torch
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:/armas/model/armas.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    detect = model(frame)

    info = detect.pandas().xyxy[0]
    print(info)

    cv2.imshow("ARMAS", np.squeeze(detect.render()))

    t = cv2.waitKey(5)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()