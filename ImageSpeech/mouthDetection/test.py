import cv2, os, sys,  dlib
from skimage import io

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

def extractFace (detector, filePath, outputPath):
    img = io.imread(filePath)
    img_width = img.shape[1]
    img_height = img.shape[0]
    dets = detector(img, 1)
    if len(dets) == 0:
        print("no faces found for file: ", filePath)
    for i, d in enumerate(dets):
        left = d.left()
        right = d.right()
        top = d.top()
        bot = d.bottom()
    # # offsets determined by testing, dependent on sizing of ffmpeg
    crop_img = img[top:bot, left:right]
    io.imsave(outputPath, crop_img)
    
    # with OPENCV
    # img = cv2.imread(filePath)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # faces = detector.detectMultiScale(gray, 1.3, 5)
    # if len(faces) > 0:
    #     (x, y, w, h) = faces[0]
    #     crop_img = img[y:y + h, x:x + w]
    #     cv2.imwrite(outputPath, crop_img)


def extractFaces (sourceDir, storeFaceDir, storeMouthDir):
    """
     detect faces, store in 'faces' folder. Extract mouth region, store in 'mouths' folder
    """
    print("extracting Faces...")

    # storeDir / videoName / VideoName_frameNumber.jpg
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(sourceDir) if isfile(join(sourceDir, f))]
    onlyfiles.sort(key=tryint)  # sorts list in place
    # print("files: ", onlyfiles)
    
    if not os.path.exists(storeFaceDir):
        os.makedirs(storeFaceDir)
    
    detector = dlib.get_frontal_face_detector()
    #detector = cv2.CascadeClassifier(
    #        '/home/matthijs/bin/anaconda/envs/convnets/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    for file in onlyfiles:
        filename, ext = os.path.splitext(file)
        videoName, frame = filename.split("_")
        filePath = ''.join([sourceDir, os.sep, file])  # the file we're processing now

        # print("extracting face and mouth from ", filePath)
        storeFacePath = ''.join([storeFaceDir, os.sep, videoName, "_face_", str(frame), ".jpg"])  # face saved here
        if not os.path.exists(storeFacePath):  # don't store if it already exists
            extractFace(detector, filePath, storeFacePath)
        
        
def extractMouth (source, target):
    img = cv2.imread(source)
    
    # simple cropping
    # h, w = img.shape[:2]
    # crop_img = img[350:520, 100:410]  # determined through testing

    # mouth detection
    mouth_cascade = cv2.CascadeClassifier('/home/matthijs/bin/anaconda/envs/convnets/share/OpenCV/haarcascades/haarcascade_mcs_nose.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mouths = mouth_cascade.detectMultiScale(gray, 1.3, 5)
    if len(mouths) > 0:
        (x, y, w, h) = mouths[0]
        crop_img = img[y:y + h, x:x + w]
    cv2.imwrite(target, crop_img)


def extractMouths(facesDir, storeMouthDir):
    print("extracting Mouths...")
    from os import listdir
    from os.path import isfile, join
    if not os.path.exists(storeMouthDir):
        os.makedirs(storeMouthDir)
    onlyfiles = [f for f in listdir(facesDir) if isfile(join(facesDir, f))]
    onlyfiles.sort(key=tryint)  # sorts list in place
    
    for file in onlyfiles:
        filename, ext = os.path.splitext(file)
        videoName, face, frame = filename.split("_")
        storeFacePath = ''.join([facesDir, os.sep, file])  # the file we're processing now

        # print("extracting face and mouth from ", filePath)
        storeMouthPath = ''.join([storeMouthDir, os.sep, videoName, "_mouth_", str(frame), ".jpg"])  # mouth saved here
        if not os.path.exists(storeMouthPath):  # don't store file if it already exists
            extractMouth(storeFacePath, storeMouthPath)
    
            
extractFaces("/home/matthijs/TCDTIMIT/processed2/volunteers/50F/sx146", "/home/matthijs/TCDTIMIT/processed2/volunteers/50F/sx146/faces","/home/matthijs/TCDTIMIT/processed2/volunteers/50F/sx146/mouths")
extractMouths("/home/matthijs/TCDTIMIT/processed2/volunteers/50F/sx146/faces","/home/matthijs/TCDTIMIT/processed2/volunteers/50F/sx146/mouths")



