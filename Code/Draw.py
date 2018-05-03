import cv2
import sys
import numpy as np

def draw_ellipse(tracked_contour,tracked_contour_image):
	ellipse = cv2.fitEllipse(tracked_contour)
	# cv2.ellipse(tracked_contour_image,ellipse,(0,255,0),2)
	cv2.imwrite("ellipse.png",tracked_contour_image)

def find_stalk(tracked_contour,tracked_contour_image):
	stalk = tuple(tracked_contour[tracked_contour[:,:,1].argmin()][0])
	# cv2.circle(tracked_contour_image,stalk,5,[255,255,255],-1)
	cv2.imwrite("find_stalk.png",tracked_contour_image)
	return stalk

def display(image,msg):
	maxsize=max(image.shape)
	scale=700/maxsize
	image=cv2.resize(image,None,fx=scale,fy=scale)
	cv2.imshow(msg,image)
	cv2.waitKey(1)
	