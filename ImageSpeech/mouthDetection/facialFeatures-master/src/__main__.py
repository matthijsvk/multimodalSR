# Importing standard libraries
import cv2
import math
import numpy as np
import sys


# Importing custom libraries
import displayUtils
import faceFeatures


# Defining the main method
def main():

	# Check if a command-line argument has been passed to the program
	if len(sys.argv) < 2:
		print 'Please specify a valid input image'
		print 'Usage: $ python __main.py__ <input_image>'
		sys.exit(-1)

	# Read the input image file
	inputImage = cv2.imread(sys.argv[1])

	# Display an error message if the argument passed is not a valid image file
	if inputImage == None:
		print 'Please specify a valid input image'
		sys.exit(-1)

	# Convert the input image to grayscale
	grayFace = cv2.cvtColor(inputImage, cv2.cv.CV_BGR2GRAY)

	# Get the rectangle bounding the left eye in the input image
	leftEye = faceFeatures.getLeftEye(grayFace)
	print 'leftEye: ', leftEye

	# Get the rectangle bounding the right eye in the input image
	rightEye = faceFeatures.getRightEye(grayFace)
	print 'rightEye: ', rightEye

	# Get the rectangle bounding the nose in the input image
	nose = faceFeatures.getNose(grayFace)
	print 'nose: ', nose

	# Get the rectangle bounding the mouth in the input image
	mouth = faceFeatures.getMouth(grayFace)
	print 'mouth: ', mouth

	# Get the rectangle bounding the eyepair in the input image
	eyepair = faceFeatures.getEyepair(grayFace)
	print 'eyepair: ', eyepair

	# Draw a rectangle on the input image around the left eye
	displayUtils.drawRectangle(inputImage, leftEye)
	# Draw a rectangle on the input image around the right eye
	displayUtils.drawRectangle(inputImage, rightEye)
	# Draw a rectangle on the input image around the nose
	displayUtils.drawRectangle(inputImage, nose)
	# Draw a rectangle on the input image around the mouth
	displayUtils.drawRectangle(inputImage, mouth)
	# Draw a rectangle on the input image around the eyepair
	displayUtils.drawRectangle(inputImage, eyepair)

	# Display the input image
	displayUtils.displayAndWait(inputImage)


# Calling the main function when the script is run
if __name__ == '__main__':

	main()
