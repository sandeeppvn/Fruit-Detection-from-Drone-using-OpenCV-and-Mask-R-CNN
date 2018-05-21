import cv2
import sys
import numpy as np
from Draw import display

'''
Direction is defined as
Distance : Forward distance (backward is negative value)
Tilt : Clockwise(right) turn degree (negative is anti-clockwise left turn)
Height : Amount of height to rise (negative indicates drop)
'''
def get_direction(window,distance,pivot):
	
	x,y = window[:2]
	pivot_x,pivot_y = pivot

	#Here 30cm is the distance the drone should be from the fruit to cut it
	Distance = distance-30

	#Tilt degree is dependent on camera scope angle, assuming scope angle is 180
	Tilt = int((pivot_x-x)*(pivot_x/180))

	#Height is the rise and fall amount and depends on the image height dimensions
	Height = pivot_y-y

	return Distance,Tilt,Height,(x,y)

def move_drone(img,direction,msg):
	cv2.putText(img, "Distance:%.2fcm   Tilt:%.2funits   Height:%.2funits  %s"%(direction[0],direction[1],direction[2],msg),(0, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 1)
	cv2.imwrite("..\Data\Results\distance.png",img)
	display(img,"Final Image")
