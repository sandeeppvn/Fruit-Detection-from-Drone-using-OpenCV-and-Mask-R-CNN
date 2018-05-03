import cv2
import sys
import numpy as np
from Distance import get_width
from Distance import get_distance_to_camera
from Distance import consistency_check
from Drone import get_direction
from Drone import move_drone
from Draw import draw_ellipse
from Draw import find_stalk


def camshift(fruit_image,fruit_window,cap):
	(x,y,w,h) = fruit_window
	x,y,w,h = int(x),int(y),int(w),int(h)
	track_window = (x,y,w,h)
	# set up the ROI for tracking
	roi = fruit_image[x:x+h, y:y+w]
	hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv_roi, np.array([0, 60,32]), np.array([180,255,255]))
	roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
	cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
	# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
	term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
	while(1):
	    ret ,frame = cap.read()
	    if ret == True:
	        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
	        # apply meanshift to get the new location
	        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
	        # Draw it on image
	        pts = cv2.boxPoints(ret)
	        pts = np.int0(pts)
	        img2 = cv2.polylines(frame,[pts],True, 255,2)
	        k = cv2.waitKey(60) & 0xff
	        if k == 27:
	            break
	        else:
	            cv2.imwrite(chr(k)+".jpg",img2)
	    else:
	        break

def apply_Hough_Transform(image):
	bilateral_filtered_image = cv2.bilateralFilter(image, 5, 175, 175)
	edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)
	# cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
	circles = cv2.HoughCircles(image,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
	circles = np.uint16(np.around(circles))
	for i in circles[0,:]:
		# draw the outer circle
		cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),2)
	return image

def get_adjusted_window_width(fruit_window_width,poly_approx_len,area,no_of_contours):
	
	#Use the parameters of the tracked_contour to determine the distance to the fruit
	
	#Set the threshold parameters
	close_contours_len,close_len,close_area = (4,0,0)
	far_contours_len,far_len,far_area = (8,0,0)

	print("Poly_Approx:",poly_approx_len)
	print("Area:",area)
	print("No of contours:",no_of_contours)

	#Aprroximating a close distance
	if((poly_approx_len > close_len) and (area > close_area) and (no_of_contours < close_contours_len)):
		fruit_window_width = fruit_window_width

	#Aprroximating a far distance
	elif((poly_approx_len > far_len) and (area > far_area) and (no_of_contours > far_contours_len)):
		fruit_window_width = fruit_window_width

	#Aprroximating a medium distance	
	else:
		fruit_window_width = fruit_window_width

	return fruit_window_width

def track_fruit(fruit_contour,fruit_window,cap,pd):

	fruit_image,fruit_contour,(approx,area,no_of_contours) = fruit_contour

	#Real Width of fruit
	fruit_width = get_width()

	print("Fruit window:\n","[x, y, w, h]:",fruit_window)

	fruit_window_width = fruit_window[2]

	#Calculate window width ,window:(x,y,w,h) and adjust window width based on params
	fruit_window_width = get_adjusted_window_width(fruit_window_width,approx,area,no_of_contours)
	
	#Get the distance to the object by tweaking parameters
	distance = get_distance_to_camera(fruit_width,fruit_window_width)

	pivot_y,pivot_x = int(fruit_image.shape[0]/2),int(fruit_image.shape[1]/2)
	cv2.circle(fruit_image,(pivot_x,pivot_y),5,[255,255,0],-1)
	cv2.imwrite("pivot_camera.png",fruit_image)
	pd_1,pd_2,pd_3 = pd

	# print(consistency_check(distance,(pd_1,pd_2,pd_3)))

	if(not(consistency_check(distance,(pd_1,pd_2,pd_3)))):

		print("Distance:",int(distance))
		direction  = get_direction(fruit_window,distance,(pivot_y,pivot_x))
		cv2.circle(fruit_image,(int(direction[-1][0]),int(direction[-1][1])),5,[0,255,0],-1)
		cv2.imwrite("pivot_image.png",fruit_image)
		print("Direction (distance,tilt,height)",direction[:-1])

		#Case 1: Far Distance (Also applicable if no_of_contours is high)
		if(distance>160 or no_of_contours>10):
			move_drone(fruit_image, direction[:-1],"Far")

		#Case 2: Close Distance
		elif(distance<=100):
			#Use canny or sobel edge detection if the fruit is close to the camera
			bilateral_filtered_image = cv2.bilateralFilter(fruit_image, 5, 175, 175)
			edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)

			# draw a fitting ellipse around the image and display it
			if(len(fruit_contour)>=5):
				tracked_contour_image = draw_ellipse(fruit_contour,fruit_image)

			
			print("Stalk co-ordinate:",find_stalk(fruit_contour,fruit_image))

			move_drone(fruit_image, direction[:-1],"Close")	

		#Case 3: Medium Distance
		else:
			#Get circles using moments or Hough Transforms when image is close and apply mask
			#fruit_image = apply_Hough_Transform(fruit_image)
			# direction = camshift(fruit_image,fruit_window,cap)
			move_drone(fruit_image,direction[:-1],"Medium")
	print()