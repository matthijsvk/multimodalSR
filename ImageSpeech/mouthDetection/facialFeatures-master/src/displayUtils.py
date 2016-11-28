# Importing libraries
import cv2
import math
import numpy as np


# Draws a rectangle on the input image
def drawRectangle(inputImage, rectangle, color=(0, 255, 0), thickness=2):

	# Note that rectangle must be of the form [x y w h]
	# x is the X coordinate top-left corner of the rectangle
	# y is the Y coordinate of the top-left corner of the rectangle
	# w is the width of the rectangle
	# h is the height of the rectangle

	cv2.rectangle(inputImage, (rectangle[0], rectangle[1]), (rectangle[0]+rectangle[2], rectangle[1]+rectangle[3]), color, thickness)


# Displays an input image and waits
def displayAndWait(inputImage, waitTime=0, windowName='image'):

	cv2.imshow(windowName, inputImage)
	cv2.waitKey(waitTime)
	cv2.destroyAllWindows()