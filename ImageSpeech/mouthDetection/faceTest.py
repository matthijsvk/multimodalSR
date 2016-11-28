import sys,os
import dlib
import cv2
from skimage import io

detector = dlib.get_frontal_face_detector()

for f in sys.argv[1:]:
    print("Processing file: {}".format(f))
    img = io.imread(f)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))

    [dir,base] = os.path.split(str(f))
    name = base.split('.')[0]
    print(name)
    outputPath = ''.join(['.',dir, os.sep, name, "_face.jpg"])
    print(outputPath)
    io.imsave(outputPath,img)

