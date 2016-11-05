# from http://stackoverflow.com/questions/10672578/extract-video-frames-in-python#10672679

# Goal: parametrized, automated version of 
#       ffmpeg -i n.mp4 -ss 00:00:20 -s 160x120 -r 1 -f singlejpeg myframe.jpg

import os, sys
from PIL import Image
import subprocess
import pickle

# 1. read all the times from the mlf file
# create a list of lists. The higher-level list contains the block of one video, the 2nd-level list contains all the lines of that video
# MLFfile
#	| video1
#		| firstPhoneme
#		| secondPhoneme
#	| video2
#	etc		
#http://stackoverflow.com/questions/3277503/how-to-read-a-file-line-by-line-into-a-list-with-python#3277516

def readfile(filename):
	with open(filename, "r") as ins:
	    array = [[]]
	    video_index = 0
	    for line in ins:
   		line = line.strip('\n') # strip newlines
		if len(line)>1: # don't save the dots lines
			if ".mp4" not in line:
				array[video_index].append(line)	
			else: #create new 2nd-level list, now store there
				array.append([line])
				video_index += 1

	return array[1:]


# process each line: split at tabs
# this gets alist from readfile, and outputs a list of times where the video should be converted to an image, 
#			 	 as well as  a list of the phonemes at those times (for keeping track of which image belongs to what)
# other inputs: timeModifier = extract image at beginning, middle, end of phoneme interval? 
#				(value between 0 and 1, 0 meaning at beginning)
def processVideoFile(videoPhonemeList, timeModifier=0.5):
	# TODO filter the non-phoneme lines (video name; end line (just a dot)
	
	videoPath =  str(videoPhonemeList[0]).replace('"','')
	phonemes = [] #list of tuples, 1= time. 2=phoneme

	for line in videoPhonemeList[1:]: #skip the path line; then just three columns with tabs in between
		splittedLine = line.split() #split on whitespaces
		start = float(splittedLine[0])/10000000
		end = float(splittedLine[1])/10000000
		extractionTime = start* (1-timeModifier) + (timeModifier)*end

		phoneme = splittedLine[2]
		phonemes.append( (extractionTime, phoneme) ) #add the tuple

	return videoPath, phonemes

def writePhonemesToFile(videoPath, phonemes, targetDir):
	videoName = os.path.basename(videoPath)		
	videoName = os.path.splitext(videoName)[0] # remove extension
	phonemeFileName = videoName + "_PHN"

	thefile = open(targetDir + os.sep + phonemeFileName, 'w')
	for item in phonemes:
  		thefile.write(' '.join(map(str,item)) + "\r\n")
	thefile.close()
	return




# 2. extract the images from the video giving them a nice name `videoName_timestamp_phoneme`
def videoToImages(videoPath, phonemes, targetDir, targetSize='160x120'):
	print("Processing: " + videoPath)
	
	for i in range(len(phonemes)):
		time = phonemes[i][0]
		phoneme = phonemes[i][1]
		size = targetSize

		fixedTime = "{0:.3f}".format(time) # for proper filenames, without dots
		fixedTime = str(fixedTime).replace('.','-')
		videoName = os.path.basename(videoPath)		
		videoName = os.path.splitext(videoName)[0] # remove extension
		outputPath= ''.join([targetDir, os.sep, videoName, "_", fixedTime, "_", phoneme, ".jpg"]) #eg vid1_00-00-01-135_sh.jpg

		print(outputPath)
		# from https://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/
		if not os.path.exists(targetDir): os.makedirs(targetDir)
		command = ['ffmpeg',
			    '-ss', "00:00:"+str(time),
			    '-i', videoPath,
			    '-frames:v','1',
			    outputPath]

		# actually run the command, only show stderror on terminal, close the processes (don't wait for user input)
		FNULL = open(os.devnull, 'w')
		p = subprocess.Popen(command, stdout=FNULL,stderr=subprocess.STDOUT, close_fds=True) #stdout=subprocess.PIPE




def processMLF(MLFfile, storeDir):
	videos = readfile('lipspeaker.mlf')
	for video in videos:
		videoPath, phonemes = processVideoFile(video)
		writePhonemesToFile(videoPath, phonemes, storeDir)
		videoToImages(videoPath, phonemes, storeDir, targetSize='160x120')
# Testing

#test=readfile('lipspeaker.mlf')
#print(len(test))
#print(len(test[0]))
#print(test[0])
#video0,phoneme0=processVideoFile(test[0])
#print(video0)
#print(phoneme0)
processMLF('lipspeaker.mlf', '/home/matthijs/Pictures/TCD-TIMIT')



