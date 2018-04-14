# USAGE
# python distance_to_camera.py

# import the necessary packages
import numpy as np
import cv2


def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	# edged = cv2.Canny(gray, 35, 125)

	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	ret,thresh = cv2.threshold(gray,127,255,0)
	# contours,hierarchy = cv2.findContours(thresh, 1, 2)
	_,contours,hierarchy= cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(thresh,contours, -1, (0, 0, 255), 1)
	res = cv2.resize(thresh,None,fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)
	cv2.imshow("t",res)
	c = sorted(contours,key=cv2.contourArea)[-2]
	# c = max(contours, key = cv2.contourArea)

	# compute the bounding box of the of the paper region and return it
	return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

# initialize the known distance from the camera to the object, which
# in this case is 45 cm
KNOWN_DISTANCE = 15.0

# initialize the known object width, which in this case, the piece of
# paper is 7 cm wide
KNOWN_WIDTH = 5

# initialize the list of images that we'll be using
IMAGE_PATHS = ["images/Distance15.jpg", "images/Distance37.jpg","images/Distance115.jpg"]

# load the first image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
image = cv2.imread(IMAGE_PATHS[0])
marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

# loop over the images
for imagePath in IMAGE_PATHS:
	# load the image, find the marker in the image, then compute the
	# distance to the marker from the camera
	image = cv2.imread(imagePath)
	marker = find_marker(image)
	cm = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

	# draw a bounding box around the image and display it
	box = np.int0(cv2.boxPoints(marker))
	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
	cv2.putText(image, "%.2fcm"%(cm),
		(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
		2.0, (0, 255, 0), 3)
	res = cv2.resize(image,None,fx=0.20, fy=0.20, interpolation = cv2.INTER_CUBIC)
	cv2.imshow("image", res)
	cv2.waitKey(0)
