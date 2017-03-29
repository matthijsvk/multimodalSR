#### help functions
from __future__ import print_function

import argparse
# remove without complaining
import os
import os.path
import sys
import traceback

import dlib
from skimage import img_as_ubyte, io
from skimage.color import rgb2gray
from skimage.transform import resize

parser = argparse.ArgumentParser(description="Preprocessing image")
add_arg = parser.add_argument
add_arg("-i", "--input_image", help="Input image")
args = parser.parse_args()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")


def detectMouth(imagePath):
    dets = []
    fname, ext = os.path.splitext(os.path.basename(imagePath))
    f = imagePath
    if ext == ".jpg":
        try:
            # print(f)
            facePath = "testImages" + os.sep + fname + "_face.jpg"
            mouthPath = "testImages" + os.sep + fname + "_mouth.jpg"

            img = io.imread(f, as_grey=True)
            width, height = img.shape[:2]

            # detect face, then keypoints. Store face and mouth
            # resize with factor 4 to increase detection speed
            resizer = 4
            dim = (int(width / resizer), int(height / resizer))
            imgSmall = resize(img, dim)
            imgSmall = img_as_ubyte(imgSmall)

            dets = detector(imgSmall, 1)  # detect face
            if len(dets) == 0:
                # print("looking on full-res image...")
                resizer = 1
                dim = (int(width / resizer), int(height / resizer))
                imgSmall = resize(img, dim)
                imgSmall = img_as_ubyte(imgSmall)

                dets = detector(imgSmall, 1)
                if len(dets) == 0:
                    print(f)
                    print("still no faces found. Using previous face coordinates...")
                    if 'top' in locals():  # could be issue if no face in first image ? #TODO
                        face_img = img[top:bot, left:right]
                        io.imsave(facePath, face_img)
                        mouth_img = img[my:my + mh, mx:mx + mw]
                        io.imsave(mouthPath, mouth_img)
                    else:
                        print("top not in locals. ERROR")

            d = dets[0]
            # extract face, store in storeFacesDir
            left = d.left() * resizer
            right = d.right() * resizer
            top = d.top() * resizer
            bot = d.bottom() * resizer
            # go no further than img borders
            if (left < 0):      left = 0
            if (right > width): right = width
            if (top < 0):       top = 0
            if (bot > height):  bot = height
            face_img = img[top:bot, left:right]
            io.imsave(facePath, face_img)  # save face image

            # now detect mouth landmarks
            # detect 68 keypoints, see dlibLandmarks.png
            shape = predictor(imgSmall, d)
            # Get the mouth landmarks.
            mx = shape.part(48).x * resizer
            mw = shape.part(54).x * resizer - mx
            my = shape.part(31).y * resizer
            mh = shape.part(57).y * resizer - my
            # go no further than img borders
            if (mx < 0):       mx = 0
            if (mw > width):   mw = width
            if (my < 0):       my = 0
            if (mh > height):  mh = height

            # scale them to get a better image of the mouth
            widthScalar = 1.5
            heightScalar = 1
            mx = int(mx - (widthScalar - 1) / 2.0 * mw)
            # my = int(my - (heightScalar - 1)/2.0*mh) #not needed, we already have enough nose
            mw = int(mw * widthScalar)
            mh = int(mh * widthScalar)

            mouth_img = img[my:my + mh, mx:mx + mw]
            io.imsave(mouthPath, mouth_img)
            return mouthPath
        except:
            print("Unexpected error:", sys.exc_info()[0])
            print(traceback.format_exc())
    return -1


def resize_image(filePath, filePathResized, keepAR=False, width=120.0):
    im = io.imread(filePath)
    if keepAR:  # Aspect Ratio
        r = width / im.shape[1]
        dim = (int(im.shape[0] * r), int(width))
        im_resized = resize(im, dim)
    else:
        im_resized = resize(im, (120, 120))
    io.imsave(filePathResized, im_resized)


def convertToGrayscale(oldFilePath, newFilePath):
    img_gray = rgb2gray(io.imread(oldFilePath))
    io.imsave(newFilePath, img_gray)  # don't write to disk if already exists
    return newFilePath


if __name__ == "__main__":
    print("Compiling functions...")
    mouthPath = detectMouth(args.input_image)  # expects npz model
    grayMouthPath = convertToGrayscale(mouthPath, mouthPath)
    resize_image(grayMouthPath, grayMouthPath)
