# Importing libraries
import cv2
import math
import numpy as np


# Specifying the location of the cascade classifiers
eyeClassifier1 = 'classifiers/haarcascade_eye.xml'
eyeClassifier2 = 'classifiers/haarcascade_mcs_lefteye.xml'
eyeClassifier3 = 'classifiers/haarcascade_lefteye_2splits.xml'
noseClassifier = 'classifiers/haarcascade_mcs_nose.xml'
mouthClassifier = 'classifiers/haarcascade_mouth.xml'
eyepairClassifier = 'classifiers/haarcascade_mcs_eyepair_small.xml'


# Get the left eye from an image of the detected face
def getLeftEye(grayFace):

	# Check if the image is invalid
	if grayFace == None:
		print 'getLeftEye(grayFace): grayFace is not a valid input image'
		return []

	# Check if the image is grayscale. If it isn't convert it to one.
	if len(grayFace.shape) == 3:
		grayFace = cv2.cvtColor(grayFace, cv2.cv.CV_COLOR_BGR2GRAY)

	# Get the dimensions of the image
	[rows, cols] = grayFace.shape

	# Perform histogram equalization on the grayscale image
	grayFace = cv2.equalizeHist(grayFace)

	# Determine the bounding box that contains the left eye
	EYE_SX = 0.16
	EYE_SY = 0.26
	EYE_SW = 0.30
	EYE_SH = 0.28

	leftX = cv2.cv.Round(cols * EYE_SX)
	topY = cv2.cv.Round(rows * EYE_SY)
	widthX = cv2.cv.Round(cols * EYE_SW)
	heightY = cv2.cv.Round(rows * EYE_SH)

	# Extract that bounding box from the grayscale image
	leftEyeRegion = grayFace[topY:topY+heightY, leftX:leftX+widthX]

	# Load the classifier
	leftEyeCascade = cv2.CascadeClassifier(eyeClassifier1)

	# Perform eye detection on the region within the bounding box
	leftEye = leftEyeCascade.detectMultiScale(leftEyeRegion, scaleFactor = 1.2, minNeighbors = 20, minSize = (20, 20), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

	# If an eye is detected return the result
	if leftEye != ():
		
		leftEye = leftEye[0]
		leftEye[0] += leftX
		leftEye[1] += topY
		return leftEye

	# Determine the bounding box for classifier 2
	EYE_SX = 0.12
	EYE_SY = 0.17
	EYE_SW = 0.37
	EYE_SH = 0.36

	leftX = cv2.cv.Round(cols * EYE_SX)
	topY = cv2.cv.Round(rows * EYE_SY)
	widthX = cv2.cv.Round(cols * EYE_SW)
	heightY = cv2.cv.Round(rows * EYE_SH)

	# Extract that bounding box from the grayscale image
	leftEyeRegion = grayFace[topY:topY+heightY, leftX:leftX+widthX]

	# Load the classifier
	leftEyeCascade = cv2.CascadeClassifier(eyeClassifier2)

	# Perform eye detection on the region within the bounding box
	leftEye = leftEyeCascade.detectMultiScale(leftEyeRegion, scaleFactor = 1.2, minNeighbors = 20, minSize = (20, 20), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

	# If an eye is detected return the result
	if leftEye != ():
		
		leftEye = leftEye[0]
		leftEye[0] += leftX
		leftEye[1] += topY
		return leftEye

	# Determine the bounding box for classifier 3
	EYE_SX = 0.12
	EYE_SY = 0.17
	EYE_SW = 0.37
	EYE_SH = 0.36

	leftX = cv2.cv.Round(cols * EYE_SX)
	topY = cv2.cv.Round(rows * EYE_SY)
	widthX = cv2.cv.Round(cols * EYE_SW)
	heightY = cv2.cv.Round(rows * EYE_SH)

	# Extract that bounding box from the grayscale image
	leftEyeRegion = grayFace[topY:topY+heightY, leftX:leftX+widthX]

	# Load the classifier
	leftEyeCascade = cv2.CascadeClassifier(eyeClassifier3)

	# Perform eye detection on the region within the bounding box
	leftEye = leftEyeCascade.detectMultiScale(leftEyeRegion, scaleFactor = 1.2, minNeighbors = 20, minSize = (20, 20), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

	# If an eye is detected return the result
	if leftEye != ():
		
		leftEye = leftEye[0]
		leftEye[0] += leftX
		leftEye[1] += topY
		return leftEye

	# If the left eye is not detected yet, relax the constraints and detect again
	leftEye = leftEyeCascade.detectMultiScale(leftEyeRegion, scaleFactor = 1.2, minNeighbors = 6, minSize = (20, 20), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

	# If an eye is detected return the result
	if leftEye != ():
		
		leftEye = leftEye[0]
		leftEye[0] += leftX
		leftEye[1] += topY
		return leftEye

	# If an eye is still not detected, relax the constraints further and detect again
	leftEye = leftEyeCascade.detectMultiScale(leftEyeRegion, scaleFactor = 1.2, minNeighbors = 1, minSize = (20, 20), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

	# If an eye is detected return the result
	if leftEye != ():
		
		leftEye = leftEye[0]
		leftEye[0] += leftX
		leftEye[1] += topY
		return leftEye

	# Even then, if an eye is not detected, return the heuristic left eye rectangle
	leftEye = [leftX, topY, widthX, heightY]


# Get the right eye from an image of the detected face
def getRightEye(grayFace):

	# Check if the image is invalid
	if grayFace == None:
		print 'getRightEye(grayFace): grayFace is not a valid input image'
		return []

	# Check if the image is grayscale. If it isn't convert it to one.
	if len(grayFace.shape) == 3:
		grayFace = cv2.cvtColor(grayFace, cv2.cv.CV_COLOR_BGR2GRAY)

	# Get the dimensions of the image
	[rows, cols] = grayFace.shape

	# Perform histogram equalization on the grayscale image
	grayFace = cv2.equalizeHist(grayFace)

	# Determine the bounding box that contains the left eye
	EYE_SX = 0.16
	EYE_SY = 0.26
	EYE_SW = 0.30
	EYE_SH = 0.28

	rightX = cv2.cv.Round(cols * (1.0 - EYE_SX - EYE_SW))
	topY = cv2.cv.Round(rows * EYE_SY)
	widthX = cv2.cv.Round(cols * EYE_SW)
	heightY = cv2.cv.Round(rows * EYE_SH)

	# Extract that bounding box from the grayscale image
	rightEyeRegion = grayFace[topY:topY+heightY, rightX:rightX+widthX]

	# Load the classifier
	rightEyeCascade = cv2.CascadeClassifier(eyeClassifier1)

	# Perform eye detection on the region within the bounding box
	rightEye = rightEyeCascade.detectMultiScale(rightEyeRegion, scaleFactor = 1.2, minNeighbors = 20, minSize = (20, 20), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

	# If an eye is detected return the result
	if rightEye != ():
		
		rightEye = rightEye[0]
		rightEye[0] += rightX
		rightEye[1] += topY
		return rightEye

	# Determine the bounding box for classifier 2
	EYE_SX = 0.12
	EYE_SY = 0.17
	EYE_SW = 0.37
	EYE_SH = 0.36

	rightX = cv2.cv.Round(cols * (1.0 - EYE_SX - EYE_SW))
	topY = cv2.cv.Round(rows * EYE_SY)
	widthX = cv2.cv.Round(cols * EYE_SW)
	heightY = cv2.cv.Round(rows * EYE_SH)

	# Extract that bounding box from the grayscale image
	rightEyeRegion = grayFace[topY:topY+heightY, rightX:rightX+widthX]

	# Load the classifier
	rightEyeCascade = cv2.CascadeClassifier(eyeClassifier2)

	# Perform eye detection on the region within the bounding box
	rightEye = rightEyeCascade.detectMultiScale(rightEyeRegion, scaleFactor = 1.2, minNeighbors = 20, minSize = (20, 20), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

	# If an eye is detected return the result
	if rightEye != ():
		
		rightEye = rightEye[0]
		rightEye[0] += rightX
		rightEye[1] += topY
		return rightEye

	# Determine the bounding box for classifier 3
	EYE_SX = 0.12
	EYE_SY = 0.17
	EYE_SW = 0.37
	EYE_SH = 0.36

	rightX = cv2.cv.Round(cols * (1.0 - EYE_SX - EYE_SW))
	topY = cv2.cv.Round(rows * EYE_SY)
	widthX = cv2.cv.Round(cols * EYE_SW)
	heightY = cv2.cv.Round(rows * EYE_SH)

	# Extract that bounding box from the grayscale image
	rightEyeRegion = grayFace[topY:topY+heightY, rightX:rightX+widthX]

	# Load the classifier
	rightEyeCascade = cv2.CascadeClassifier(eyeClassifier3)

	# Perform eye detection on the region within the bounding box
	rightEye = rightEyeCascade.detectMultiScale(rightEyeRegion, scaleFactor = 1.2, minNeighbors = 20, minSize = (20, 20), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

	# If an eye is detected return the result
	if rightEye != ():
		
		rightEye = rightEye[0]
		rightEye[0] += rightX
		rightEye[1] += topY
		return rightEye

	# If the left eye is not detected yet, relax the constraints and detect again
	rightEye = rightEyeCascade.detectMultiScale(rightEyeRegion, scaleFactor = 1.1, minNeighbors = 6, minSize = (20, 20), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

	# If an eye is detected return the result
	if rightEye != ():
		
		rightEye = rightEye[0]
		rightEye[0] += rightX
		rightEye[1] += topY
		return rightEye

	# If an eye is still not detected, relax the constraints further and detect again
	rightEye = rightEyeCascade.detectMultiScale(leftEyeRegion, scaleFactor = 1.1, minNeighbors = 1, minSize = (20, 20), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

	# If an eye is detected return the result
	if rightEye != ():
		
		rightEye = rightEye[0]
		rightEye[0] += rightX
		rightEye[1] += topY
		return rightEye

	# Even then, if an eye is not detected, return the heuristic right eye rectangle
	rightEye = [rightX, topY, widthX, heightY]


# Get the nose from the image of the detected face
def getNose(grayFace):

	# Check if the image is invalid
	if grayFace == None:
		print 'getRightEye(grayFace): grayFace is not a valid input image'
		return []

	# Check if the image is grayscale. If it isn't convert it to one.
	if len(grayFace.shape) == 3:
		grayFace = cv2.cvtColor(grayFace, cv2.cv.CV_COLOR_BGR2GRAY)

	# Get the dimensions of the image
	[rows, cols] = grayFace.shape

	# Perform histogram equalization on the grayscale image
	grayFace = cv2.equalizeHist(grayFace)

	# Determine the bounding box that contains the nose
	TOP_Y = 0.40
	BOTTOM_Y = 0.80
	LEFT_X = 0.31
	RIGHT_X = 0.69

	topY = cv2.cv.Round(rows * TOP_Y)
	bottomY = cv2.cv.Round(rows * BOTTOM_Y)
	leftX = cv2.cv.Round(cols * LEFT_X)
	rightX = cv2.cv.Round(cols * RIGHT_X)

	# Extract that bounding box from the grayscale image
	noseRegion = grayFace[topY:bottomY, leftX:rightX]

	# Load the classifier
	noseCascade = cv2.CascadeClassifier(noseClassifier)

	# Perform nose detection on the region within the bounding box
	nose = noseCascade.detectMultiScale(noseRegion, scaleFactor = 1.2, minNeighbors = 6, minSize = (20, 20), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

	# If a nose is detected return the result
	if nose != ():
		
		nose = nose[0]
		nose[0] += leftX
		nose[1] += topY
		return nose

	# If a nose is not detected, lower the threshold and try again
	nose = noseCascade.detectMultiScale(noseRegion, scaleFactor = 1.2, minNeighbors = 3, minSize = (20, 20), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

	# If a nose is detected return the result
	if nose != ():
		
		nose = nose[0]
		nose[0] += leftX
		nose[1] += topY
		return nose

	# If a nose is still not detected, lower the threshold even more and try again
	nose = noseCascade.detectMultiScale(noseRegion, scaleFactor = 1.1, minNeighbors = 1, minSize = (20, 20), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

	# Even then, if a nose isn't detected, return an empty tuple
	return []


# Get the mouth from the image of the detected face
def getMouth(grayFace):

	# Check if the image is invalid
	if grayFace == None:
		print 'getRightEye(grayFace): grayFace is not a valid input image'
		return []

	# Check if the image is grayscale. If it isn't convert it to one.
	if len(grayFace.shape) == 3:
		grayFace = cv2.cvtColor(grayFace, cv2.cv.CV_COLOR_BGR2GRAY)

	# Get the dimensions of the image
	[rows, cols] = grayFace.shape

	# Perform histogram equalization on the grayscale image
	grayFace = cv2.equalizeHist(grayFace)

	# Determine the bounding box that contains the nose
	TOP_Y = 0.67
	BOTTOM_Y = 0.98
	LEFT_X = 0.2
	RIGHT_X = 0.8

	topY = cv2.cv.Round(rows * TOP_Y)
	bottomY = cv2.cv.Round(rows * BOTTOM_Y)
	leftX = cv2.cv.Round(cols * LEFT_X)
	rightX = cv2.cv.Round(cols * RIGHT_X)

	# Extract that bounding box from the grayscale image
	mouthRegion = grayFace[topY:bottomY, leftX:rightX]

	# Load the classifier
	mouthCascade = cv2.CascadeClassifier(mouthClassifier)

	# Perform mouth detection on the region within the bounding box
	mouth = mouthCascade.detectMultiScale(mouthRegion, scaleFactor = 1.2, minNeighbors = 6, minSize = (20, 20), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

	# If a mouth is detected return the result
	if mouth != ():
		
		mouth = mouth[0]
		mouth[0] += leftX
		mouth[1] += topY
		return mouth

	# If a mouth is not detected, lower the threshold and try again
	mouth = mouthCascade.detectMultiScale(mouthRegion, scaleFactor = 1.2, minNeighbors = 3, minSize = (20, 20), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

	# If a mouth is detected return the result
	if mouth != ():
		
		mouth = mouth[0]
		mouth[0] += leftX
		mouth[1] += topY
		return mouth

	# If a mouth is still not detected, lower the threshold even more and try again
	mouth = mouthCascade.detectMultiScale(mouthRegion, scaleFactor = 1.1, minNeighbors = 1, minSize = (20, 20), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

	# If a mouth is detected return the result
	if mouth != ():
		
		mouth = mouth[0]
		mouth[0] += leftX
		mouth[1] += topY
		return mouth

	# Even then, if a mouth isn't detected, return an empty tuple
	return []


# Get the eyepair from the image of the detected face
def getEyepair(grayFace):

	# Check if the image is invalid
	if grayFace == None:
		print 'getRightEye(grayFace): grayFace is not a valid input image'
		return []

	# Check if the image is grayscale. If it isn't convert it to one.
	if len(grayFace.shape) == 3:
		grayFace = cv2.cvtColor(grayFace, cv2.cv.CV_COLOR_BGR2GRAY)

	# Get the dimensions of the image
	[rows, cols] = grayFace.shape

	# Perform histogram equalization on the grayscale image
	grayFace = cv2.equalizeHist(grayFace)

	# Determine the bounding box that contains the nose
	TOP_Y = 0.15
	BOTTOM_Y = 0.55
	LEFT_X = 0.10
	RIGHT_X = 0.90

	topY = cv2.cv.Round(rows * TOP_Y)
	bottomY = cv2.cv.Round(rows * BOTTOM_Y)
	leftX = cv2.cv.Round(cols * LEFT_X)
	rightX = cv2.cv.Round(cols * RIGHT_X)

	# Extract that bounding box from the grayscale image
	eyepairRegion = grayFace[topY:bottomY, leftX:rightX]

	# Load the classifier
	eyepairCascade = cv2.CascadeClassifier(eyepairClassifier)

	# Perform eyepair detection on the region within the bounding box
	eyepair = eyepairCascade.detectMultiScale(eyepairRegion, scaleFactor = 1.2, minNeighbors = 6, minSize = (20, 20), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

	# If an eyepair is detected return the result
	if eyepair != ():
		
		eyepair = eyepair[0]
		eyepair[0] += leftX
		eyepair[1] += topY
		return eyepair

	# If an eyepair is not detected, lower the threshold and try again
	eyepair = eyepairCascade.detectMultiScale(eyepairRegion, scaleFactor = 1.2, minNeighbors = 3, minSize = (20, 20), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

	# If an eyepair is detected return the result
	if eyepair != ():
		
		eyepair = eyepair[0]
		eyepair[0] += leftX
		eyepair[1] += topY
		return eyepair

	# If an eyepair is still not detected, lower the threshold even more and try again
	eyepair = eyepairCascade.detectMultiScale(eyepairRegion, scaleFactor = 1.1, minNeighbors = 1, minSize = (20, 20), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

	# If an eyepair is detected return the result
	if eyepair != ():
		
		eyepair = eyepair[0]
		eyepair[0] += leftX
		eyepair[1] += topY
		return eyepair

	# Even then, if a mouth isn't detected, return an empty tuple
	return []