from __future__ import print_function
import os,sys
import numpy as np
from PIL import Image

def getPhonemeNumberMap(phonemeMap="/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/Thesis/ImageSpeech/phonemeLabelConversion.txt"):
    phonemeNumberMap = {}
    with open(phonemeMap) as inf:
        for line in inf:
            parts = line.split()  # split line into parts
            if len(parts) > 1:  # if at least 2 parts/columns
                phonemeNumberMap[str(parts[0])] = parts[1]  # part0= frame, part1 = phoneme
                phonemeNumberMap[str(parts[1])] = parts[0]
    return phonemeNumberMap


rootDir = "/home/matthijs/TCDTIMIT_test/database/Volunteers/03F"
targetDir = "/home/matthijs/TCDTIMIT_test/database_binary"
if not os.path.exists(targetDir):
    os.makedirs(targetDir)
    
# get list of images and list of labels
images = []
labels = []
for root, dirs, files in os.walk(rootDir):
        for file in files:
            name, extension = os.path.splitext(file)
            # copy phoneme files as well
            if extension == ".jpg":
                videoName, frame, phoneme = name.split("_")
                path = ''.join([root, os.sep, file])
                print(path, " is \t ", phoneme)
                images.append(path)
                labels.append(phoneme)

# write label and image to binary file, 1 label+image per row
videoName = os.path.basename(rootDir)
output_filename = targetDir+os.sep+videoName+".bin"
print(output_filename)

with open(output_filename, "wb") as f: # from http://stackoverflow.com/questions/38880654/how-do-i-create-a-dataset-with-multiple-images-the-same-format-as-cifar10?rq=1
  for label, img in zip(labels, images):
    
    phonemeNumberMap = getPhonemeNumberMap()
    labelNumber = phonemeNumberMap[label]
    npLabel = np.array(labelNumber, dtype=np.uint8)
    
    f.write(npLabel.tostring())  # Write label.
    im = np.array(Image.open(img), dtype=np.uint8)
    f.write(im[:, :].tostring())  # Write grey channel, it's the only one
    

