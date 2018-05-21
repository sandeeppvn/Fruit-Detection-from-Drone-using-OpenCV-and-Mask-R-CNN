import cv2
import sys
import numpy as np
import math

def get_width():

	apple = 7.5
	return apple

def get_distance_to_camera(knownWidth, perWidth):

	#Obtain focalLength from training or direct specs
	focalLength = 1200r
	# compute and return the distance from the maker to the camera
	distance = (knownWidth * focalLength) / perWidth
	return distance

def get_distance(p1,p2):

	return int(math.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2  ))

def consistency_check(distance,pd):
	
	pd_1,pd_2,pd_3 = pd
	if(pd_1==0):
		pd_1 = distance

	elif(not(pd_1==0) and pd_2==0):
		pd_2 = pd_1
		pd_1 = distance

	elif(not(pd_1==0) and not(pd_2==0)):
		pd_3 = pd_2
		pd_2 = pd_1
		pd_1 = distance
	
	if( not(pd_1==0) and not(pd_1==0) and not(pd_2==0) and not(pd_3==0) and (abs(pd_1-pd_2)<10 or abs(pd_2-pd_3)<10 or abs(pd_3-pd_1)<10)):
		return True
	else:
		return False
