### help functions

# remove without complaining
import os, errno


def silentremove (filename):
    try:
        os.remove(filename)
    except OSError as e:  # name the Exception `e`
        pass #print("Failed with:", e.strerror)  # look what it says


# sort based on filenames, numerically instead of lexicographically
def tryint (s):
    t = os.path.splitext(s)[0]
    try:
        return int(t)
    except:
        try:
            u = t.split("_")[1]
            return int(u)
        except:
            return t
                       
# extract faces out of an image using dlib
import sys, os
import dlib
from skimage import io
import cv2

def extractFace (filePath, outputPath):
    detector = dlib.get_frontal_face_detector()
    img = io.imread(filePath)
    
    img_width = img.shape[1]
    img_height = img.shape[0]
    
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(img, 1)
    for i, d in enumerate(dets):
        left = d.left()
    right = d.right()
    top = d.top()
    bot = d.bottom()
    
    # increase size of rectangle
    factor = 0.5;
    add_width = 0  # int( factor/2.0 * abs(right - left))
    add_height = int(factor / 2.0 * abs(top - bot))
    
    if (top > add_height):
        top -= add_height
    else:
        top = 0
    if (bot + add_height < img_height):
        bot += add_height
    else:
        bot = img_height
    
    if (left > add_width):
        left -= add_width
    else:
        left = 0
    if (right + add_width < img_width):
        right += add_width
    else:
        right = img_width
    
    top = int(top + abs((top - bot) / 2.0))
    
    crop_img = img[top:bot, left:right]
    io.imsave(outputPath, crop_img)

    # increase size of rectangle
    factor = 0.5;
    add_width = 0  # int( factor/2.0 * abs(right - left))
    add_height = int(factor / 2.0 * abs(top - bot))


#
