import numpy as np
from PIL import Image
imsize = 120*120
data= np.zeros(shape=(2, imsize), dtype=np.uint8)

img = "/home/matthijs/TCDTIMIT/database/lipspeakers/Lipspkr1/sa1_37_iy.jpg"
im = np.array(Image.open(img), dtype=np.uint8).flatten()
data[0] = im

img2 = "/home/matthijs/TCDTIMIT/database/lipspeakers/Lipspkr1/sa1_34_sh.jpg"
im2 = np.array(Image.open(img2), dtype=np.uint8).flatten()
data[1] = im2

print(im2)
print(len(im2))


print(im.shape)
print(im2.shape)
print(data.shape)

print(data[1])

print(type(data[1][0]))
print(type(im2[0]))
