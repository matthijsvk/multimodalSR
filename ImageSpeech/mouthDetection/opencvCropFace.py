import numpy as np
import cv2
import sys, os
# from http://www.docs.opencv.org/master/d7/d8b/tutorial_py_face_detection.html



# these aren't robust, don't use them
# mouth_cascade = cv2.CascadeClassifier('/home/matthijs/bin/anaconda/envs/convnets/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml')
# eye_cascade = cv2.CascadeClassifier('/home/matthijs/bin/anaconda/envs/convnets/share/OpenCV/haarcascades/haarcascade_mcs_eyepair_big.xml')

face_cascade = cv2.CascadeClassifier('/home/matthijs/bin/anaconda/envs/convnets/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

def cropFaces(dirPath):
    from os import listdir
    from os.path import isfile, join
    images = [f for f in listdir(dirPath) if isfile(join(dirPath, f))]
    print(images)
    for imgName in images:
        if "cropped" in imgName:
            continue
        imgPath = dirPath + os.sep + imgName
        img = cv2.imread(imgPath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            crop_img = img[y:y + h, x:x + w]
            imgPath  = os.path.splitext(imgPath)[0]
            cv2.imwrite(imgPath+"cropped.jpg",crop_img)

            def extractMouth (source, target):
                img = cv2.imread(source)
                h, w = img.shape[:2]
                crop_img = img[350:520, 100:410]  # determined through testing
                cv2.imwrite(target, crop_img)
            #cv2.imwrite(imgPath, crop_img) # overwrite the old image

cropFaces("./faces/sx146")
#cropFaces("/home/matthijs/TCDTIMIT/processed2/volunteers/50F/sx146")
##
# mouths = mouth_cascade.detectMultiScale(roi_gray)
#         print(mouths)
#         for (mx, my, mw, mh) in mouths:
#             mx = int(mx*1)
#             my = int(my*1)
#             mw = int(mw*1)
#             mh = int(mh*1)
#             cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 255, 0), 2)
#             crop_img = img[mx:mx+mw, my:my+mh] #img[y: y + h, x: x + w]
#             print(imgPath)
#             imgPath  = os.path.splitext(imgPath)[0]
#             cv2.imwrite(imgPath+"cropped.jpg",crop_img)