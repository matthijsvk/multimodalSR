#!/usr/bin/python
import sys
import os
import dlib
import glob
from skimage import io
import cv2


# detect faces in all jpg's in sourceDir
# extract faces to "storeDir/faces", and mouths to "storeDir/mouths"
def extractFacesMouths (sourceDir, storeDir):
    import dlib
    storeFaceDir = storeDir + os.sep + "faces"
    if not os.path.exists(storeFaceDir):
        os.makedirs(storeFaceDir)
    
    storeMouthsDir = storeDir + os.sep + "mouths"
    if not os.path.exists(storeMouthsDir):
        os.makedirs(storeMouthsDir)
    
    detector = dlib.get_frontal_face_detector()
    predictor_path = "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/Thesis/ImageSpeech/mouthDetection/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    
    for f in glob.glob(os.path.join(sourceDir, "*.jpg")):
        fname = os.path.splitext(os.path.basename(f))[0]
        facePath = storeFaceDir + os.sep + fname + "_face.jpg"
        mouthPath = storeMouthsDir + os.sep + fname + "_mouth.jpg"
        img = io.imread(f)
        
        # don't reprocess images that already have a face extracted, as we've already done this before!
        if os.path.exists(facePath): continue
        
        # detect face, then keypoints. Store face and mouth
        resizer = 4
        height,width=img.shape[:2]
        imgSmall =  cv2.resize(img,(int(width/resizer), int(height/resizer)), interpolation = cv2.INTER_AREA) #linear for zooming, inter_area for shrinking
        imgSmall =  cv2.cvtColor(imgSmall, cv2.COLOR_BGR2GRAY)
        
        dets = detector(imgSmall, 1)  # detect face, don't upsample
        if len(dets) == 0:
            print("no faces found for file: ", f)
        else:
            for i, d in enumerate(dets):
                # extract face, store in storeFacesDir
                left = d.left() * resizer
                right = d.right() * resizer
                top = d.top() * resizer
                bot = d.bottom() * resizer
                face_img = img[top:bot, left:right]
                io.imsave(facePath, face_img)  # don't write to disk if already exists
                
                # detect 68 keypoints
                shape = predictor(imgSmall, d)
                # Get the mouth landmarks.
                mx = shape.part(48).x * resizer
                mw = shape.part(54).x * resizer - mx
                my = shape.part(31).y * resizer
                mh = shape.part(57).y * resizer - my

                # scale them to get a better image
                widthScalar = 1.5
                heightScalar = 1
                mx = int(mx - (widthScalar - 1) / 2.0 * mw)
                # my = int(my - (heightScalar - 1)/2.0*mh) #not need,d we already have enough nose
                mw = int(mw * widthScalar)
                mh = int(mh * widthScalar)
                
                mouth_img = img[my:my + mh, mx:mx + mw]
                io.imsave(mouthPath, mouth_img)

import time
startTime = time.clock()
extractFacesMouths("/home/matthijs/test/processed/03F/sx77","/home/matthijs/test/processed/03F/sx77")
duration = time.clock() - startTime

print("This took ", duration, " seconds.")

