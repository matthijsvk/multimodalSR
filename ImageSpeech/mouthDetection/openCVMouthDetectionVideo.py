import cv2
import numpy as np
 # from https://www.safaribooksonline.com/library/view/opencv-with-python/9781785283932/ch04s08.html
mouth_cascade = cv2.CascadeClassifier('/home/matthijs/bin/anaconda/envs/convnets/share/OpenCV/haarcascades/haarcascade_mcs_nose.xml')
mouth_cascade2 = cv2.CascadeClassifier('/home/matthijs/bin/anaconda/envs/convnets/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml')

if mouth_cascade.empty():
  raise IOError('Unable to load the mouth cascade classifier xml file')

cap = cv2.VideoCapture(0)
ds_factor = 1

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    print("width: %d | height: %d ", height, width)
    #frame = frame[int(height/2):height, 0:width]
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
    if len(mouth_rects) > 0:
        for (x,y,w,h) in mouth_rects:
            w = int(w * 4)
            h = int(h)
            y = int(y + 0.4*h)
            x = int(x - 0.10*w)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
            break
    else:
        mouth_rects = mouth_cascade2.detectMultiScale(gray, 1.7, 11)
        if len(mouth_rects) > 0:
            for (x, y, w, h) in mouth_rects:
                #y = int(y - 0.15 * h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                break
    frame = frame[y:y+h, x:x+h]
    cv2.imshow('Mouth Detector', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
