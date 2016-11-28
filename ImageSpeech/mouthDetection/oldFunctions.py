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


