import sys, os
import dlib
from skimage import io

detector = dlib.get_frontal_face_detector()

for f in sys.argv[1:]:
    #print("Processing file: {}".format(f))
    img = io.imread(f)
    
    img_width = img.shape[1]
    img_height = img.shape[0]
    
    dets = detector(img, 1)
    #print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        left = d.left()
    right = d.right()
    top = d.top()
    bot = d.bottom()
    #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
    #        i, left, top, right, bot))

    # increase size of rectangle
    factor = 0.5;
    add_width = 0  # int( factor/2.0 * abs(right - left))
    add_height = int(factor / 2.0 * abs(top - bot))
    print(add_height)
    
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
    
    top = top + abs((top - bot) / 2.0)
    
    print("After size increase {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, left, top, right, bot))
    
    crop_img = img[top:bot, left:right]
    [dir, base] = os.path.split(str(f))
    name = base.split('.')[0]
    # print(name)
    outputPath = ''.join([dir, os.sep, name, "_face.jpg"])
    # print(outputPath)
    io.imsave(outputPath, crop_img)