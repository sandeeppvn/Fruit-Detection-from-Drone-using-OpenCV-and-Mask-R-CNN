import cv2
import sys
import numpy as np
import time
from extractFruit import extract_fruit
from TrackFruit import track_fruit

pd_1 = 0
pd_2 = 0
pd_3 = 0

if __name__ == '__main__':

	# global pd_1
	# global pd_2
	# global pd_3

	img_name = sys.argv[1]
	
	#If input is a Video
	if(len(sys.argv)<=2):

		cap = cv2.VideoCapture(img_name)
		time.sleep(5)
		ret, frame = cap.read()
					
		while(cap.isOpened() and ret):
			
			variance = cv2.Laplacian(frame, cv2.CV_64F).var()

			#Later check variance according to contour size
			val = extract_fruit(frame,variance)
			if(isinstance(val, int)):
				continue
			fruit_contour,fruit_window = val
			track_fruit(fruit_contour,fruit_window,cap,(pd_1,pd_2,pd_3))
			ret, frame = cap.read()


	#If input is an Image
	else:
		
		img = cv2.imread(img_name)
		variance = cv2.Laplacian(img, cv2.CV_64F).var()
		val = extract_fruit(frame,variance)
		if(not(isinstance(val, int))):				
			fruit_contour,fruit_window = val
			track_fruit(fruit_contour,fruit_window,1,(pd_1,pd_2,pd_3))

