def extractFaces (sourceDir, storeFaceDir, storeMouthDir, videoName):
    """
     detect faces, store in 'faces' folder. Extract mouth region, store in 'mouths' folder
    """
    # storeDir / videoName / VideoName_frameNumber.jpg
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(sourceDir) if isfile(join(sourceDir, f))]
    onlyfiles.sort(key=tryint)  # sorts list in place
    # print("files: ", onlyfiles)
    
    if not os.path.exists(storeFaceDir):
        os.makedirs(storeFaceDir)
    if not os.path.exists(storeMouthDir):
        os.makedirs(storeMouthDir)

    # detector = dlib.get_frontal_face_detector()
    detector = cv2.CascadeClassifier(
            '/home/matthijs/bin/anaconda/envs/convnets/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    for file in onlyfiles:
        filename, ext = os.path.splitext(file)
        videoName, frame = filename.split("_")
        filePath = ''.join([sourceDir, os.sep, file])  # the file we're processing now

        # print("extracting face and mouth from ", filePath)
        storeFacePath = ''.join([storeFaceDir, os.sep, videoName, "_face_", str(frame), ".jpg"])  # face saved here
        if not os.path.exists(storeFacePath):  # don't store if it already exists
            extractFace(detector, filePath, storeFacePath)
        
        storeMouthPath = ''.join([storeMouthDir, os.sep, videoName, "_mouth_", str(frame), ".jpg"])  # mouth saved here
        if not os.path.exists(storeMouthPath):  # don't store if it already exists
            extractMouth(storeFacePath, storeMouthPath)




def extractFace (detector, filePath, outputPath):
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
    factor = 0.2;
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
    
    crop_img = img[top + 30:bot - 50, left + 100:right - 100]  # offsets determined by testing
    io.imsave(outputPath, crop_img)



# extract the images at the times of the phonemes from the videom giving them a nice name `videoName_timestamp_phoneme`. Crop approximate mouth region
def videoToImages (videoPath, phonemes, targetDir, targetSize='296:224', cropStartPixel='888:614'):
    print("Processing: " + videoPath)
    
    for i in range(len(phonemes)):
        extractionTimeFloat = phonemes[i][0]
        extractionTime = str(extractionTimeFloat).replace('.', '-')  # needs this format for ffmpeg
        
        phoneme = phonemes[i][1]
        
        videoName = os.path.basename(videoPath)
        videoName = os.path.splitext(videoName)[0]  # remove extension
        outputPath = ''.join([targetDir, os.sep, videoName, "_", extractionTime, "_", phoneme,
                              ".jpg"])  # eg vid1_00-00-01-135_sh.jpg
        
        # print(outputPath)
        # from https://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/
        if not os.path.exists(targetDir): os.makedirs(targetDir)
        command = ['ffmpeg',
                   '-ss', "00:00:" + extractionTimeFloat,
                   '-i', videoPath,
                   '-s', targetSize,
                   '-vf', "crop=" + targetSize + ":" + cropStartPixel,
                   '-frames:v', '1',
                   outputPath]
        
        # actually run the command, only show stderror on terminal, close the processes (don't wait for user input)
        FNULL = open(os.devnull, 'w')
        p = subprocess.Popen(command, stdout=FNULL, stderr=subprocess.STDOUT, close_fds=True)  # stdout=subprocess.PIPE
    return 0


# extract only the valid frames from ffmpg
# result: downloads the frames, but lots of copies
def extractValidFrames (phonemes, videoPath, storeDir, framerate=29.97, targetSize='960x540'):
    import subprocess
    # 1. [extracting all frames](https://ubuntuforums.org/showthread.php?t=1141293): `mkdir videoName; ffmpeg -i VideoName.mp4 frames/%d.jpg`
    videoName = os.path.basename(videoPath)
    videoName = os.path.splitext(videoName)[0]
    targetDir = storeDir + os.sep + videoName
    if not os.path.exists(targetDir): os.makedirs(targetDir)
    outputPath = ''.join(
            [storeDir, os.sep, videoName, os.sep])  # eg vid1_. frame number and extension will be added by ffmpeg
    
    validTimes, validFrames, validPhonemes = getValid(phonemes, framerate)
    # compile the selection command, see http://stackoverflow.com/questions/38253406/extract-list-of-specific-frames-using-ffmpeg#38259151
    # should look like this: ffmpeg -i in.mp4 -vf select='eq(n\,100)+eq(n\,184)+eq(n\,213)' -vsync 0 frames%d.jpg
    selection = "select='"
    for frame in validFrames[0:-1]:
        selection = "".join([selection, r"eq(n\,", str(frame), ")+"])
    selection = "".join([selection, r"eq(n\,", str(validFrames[-1] - 1), ")'"])
    selection = selection.decode('string_escape')
    print(selection)
    
    command = ''.join(["ffmpeg",
                       " -i ", videoPath,
                       " -s ", targetSize,
                       " -vsync 0",
                       " -vf ", selection,  # .decode('string_escape'),
                       " ", outputPath, "%d.jpg"])  # add frame number and extension
    # actually run the command, only show stderror on terminal, close the processes (don't wait for user input)
    FNULL = open(os.devnull, 'w')
    p = subprocess.Popen(command, stdout=FNULL, stderr=subprocess.STDOUT, shell=True,
                         close_fds=True)  # stdout=subprocess.PIPE
    
    # now we'll have the right frames, but they will be numbered 1-nbValidFrames, not with the correct frame number
    filenames = [filename for filename in os.listdir(targetDir)]
    filenames.sort(key=tryint)  # sorts list in place
    print(len(filenames), len(validFrames))
    for i in range(len(validFrames)):
        # print(targetDir,os.sep,filenames[i])
        # print(filenames[i]," -> ",validFrames[i])
        # os.rename(targetDir+os.sep+filenames[i], ''.join([targetDir,os.sep,videoName, "_", str(validFrames[i]),".jpg"]))
        i += 1
    return 0



# this doesn't work because of the delay when writing to disk -> need to work with batches
def extractMouths (phonemes, videoPath, storeDir, framerate=29.97):
    # 1. [extracting all frames](https://ubuntuforums.org/showthread.php?t=1141293): `mkdir videoName; ffmpeg -i VideoName.mp4 frames/%d.jpg`
    # 2. calculate needed frames from video labels
    # 3. throw away all non-needed frames
    # 4. compress
    # 5. extract face
    # 6. extract mouth
    if not os.path.exists(videoPath):
        print("This video does not exist:", videoPath)
        return -1
    
    videoName = os.path.basename(videoPath)
    videoName = os.path.splitext(videoName)[0]  # remove extension
    
    if os.path.exists(storeDir + os.sep + videoName):
        print("video already processed. Skipping...")
        return 0
    
    # 1. [extracting all frames](https://ubuntuforums.org/showthread.php?t=1141293): `mkdir videoName; ffmpeg -i VideoName.mp4 frames/%d.jpg`
    # stored in 'storeDir/videoName/VideoName_frameNumber.jpg'
    extractAllFrames(videoPath, storeDir, framerate, '1920x1080')
    
    # this takes some time, extract for another video before running the next command
    
    # 2. calculate needed frames from video labels
    removeInvalidFrames(phonemes, videoName, storeDir, framerate)
    
    # 3. extract face from images
    # extractFaces(storeDir, videoName)

