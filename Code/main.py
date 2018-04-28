import cv2
import sys
import numpy as np
import math

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

def draw_ellipse(tracked_contour,tracked_contour_image):
	ellipse = cv2.fitEllipse(tracked_contour)
	cv2.ellipse(tracked_contour_image,ellipse,(0,255,0),2)


def get_window(val):

	#Make a rectangular window for the contour to track
	tracked_contour_image,tracked_contour,tracked_contour_params = val
	poly_approx_len,area,no_of_contours = tracked_contour_params

	print()
	print("Tracked Contour Fetaures:")
	print("\tNo of cont:",no_of_contours)
	print("\tPoly_approx_len:",poly_approx_len)
	print("\tArea:",area)
	

	# draw a fitting ellipse around the image and display it
	if(len(tracked_contour)>=5):
		tracked_contour_image = draw_ellipse(tracked_contour,tracked_contour_image)
		

	# draw a bounding rotated box around the image and display it
	rect = cv2.minAreaRect(tracked_contour)
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	cv2.drawContours(tracked_contour_image,[box],0,(0,0,255),2)
	
	window_width = ( get_distance(box[0],box[1]) + get_distance(box[2],box[3]) )/2.0
	window_height = ( get_distance(box[0],box[3]) + get_distance(box[1],box[2]) )/2.0

	# print("Window Width",window_width)
	# print("Window Height",window_height)

	mid_x = (box[0][0]+box[1][0]+box[2][0]+box[3][0])/4
	mid_y = (box[0][1]+box[1][1]+box[2][1]+box[3][1])/4
	window = [mid_x,mid_y,window_width,window_height]

	return window

def display(image,msg):
	maxsize=max(image.shape)
	scale=700/maxsize
	image=cv2.resize(image,None,fx=scale,fy=scale)
	cv2.imshow(msg,image)
	cv2.waitKey(x)

def find_stalk(image):

	pass

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

def get_direction(window,distance):
	#Direction is defined as
	#Distance : Forward distance (backward is negative value)
	#Tilt : Clockwise(right) turn degree (negative is anti-clockwise left turn)
	#Height : Amount of height to rise (negative indicates drop)
	x = window[0]
	y = window[1]


	#Here 30cm is the distance the drone should be from the fruit to cut it
	Distance = distance-30

	#Tilt degree is dependent on camera scope angle, assuming scope angle is 180
	degree = pivot_x/180
	Tilt = int((pivot_x-x)*degree)

	#Height is the rise and fall amount and depends on the image height dimensions
	Height = pivot_y-y

	return Distance,Tilt,Height,(x,y)

def move_drone(img,direction,msg):
	cv2.putText(img, "Distance:%.2fcm   Tilt:%.2funits   Height:%.2funits  %s"%(direction[0],direction[1],direction[2],msg),(0, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 1)
	display(img,"Final Image")
	# print("Pivot:",pivot_x,pivot_y)img
	pass

def get_contours(image):
		
		im = image.copy()
		image=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		image, contours, hierarchy = cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		# cv2.drawContours(im, contours, -1,(255,255,0), 3)
		#Down list is len(approx),hull,area
		#Here Use convex hull on the contour
		contour_list = []
		for contour in contours:
			hull = cv2.convexHull(contour)
			approx = len(cv2.approxPolyDP(hull,0.1*cv2.arcLength(hull,True),True))
			area = cv2.contourArea(hull)
			contour_list.append([hull,approx,area])

		
		contour_list.sort(key=lambda contour: contour[2],reverse=True)
		print(len(contour_list),"cnt")

		#Get the largest contour out of all the contours
		if(len(contour_list)==0):
			return 0
		if(len(contour_list)==1):
			hull,approx,area = contour_list[0]
		else:
			hull,approx,area = contour_list[1]

		tracked_contour_params = [approx,area,len(contours)]
		return im,hull,tracked_contour_params

def get_fruit(image,variance):

	def preprocessing(image):
		#PRE PROCESSING OF IMAGE

		# image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		maxsize=max(image.shape)
		scale=700/maxsize
		image=cv2.resize(image,None,fx=scale,fy=scale)

		image_blur=cv2.GaussianBlur(image,(7,7),0)
		image_blur_hsv=cv2.cvtColor(image_blur,cv2.COLOR_BGR2HSV)
		

		return image,image_blur,image_blur_hsv

	def get_color_mask(image):
		# #Strawberry
		# min_color=np.array([0,100,80])
		# max_color=np.array([10,256,256])
		# min_color2=np.array([170,100,80])
		# max_color2=np.array([180,256,256])
		#
		# #Banana
		# min_color=np.array([20,50,50])
		# max_color=np.array([30,256,256])
		# min_color2=np.array([60,50,50])
		# max_color2=np.array([70,256,256])

		#Apple
		min_color=np.array([0,100,80])
		max_color=np.array([10,256,256])
		min_color2=np.array([170,100,80])
		max_color2=np.array([180,256,256])

		# #Apple2
		# min_color=np.array([0,50,80])
		# max_color=np.array([10,256,256])
		# min_color2=np.array([170,100,80])
		# max_color2=np.array([180,256,256])



		mask1=cv2.inRange(image,min_color,max_color)
		mask2=cv2.inRange(image,min_color2,max_color2)

		mask=mask1+mask2

		# min_all = np.array([0,0,0])
		# max_all = np.array([180,256,256])
		# mask = cv2.inRange(image,min_all,max_all)
		return mask

	def apply_morphology(mask):
		# kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
		kernel = np.ones((3,3),np.uint8)
		dilation = cv2.dilate(mask,kernel,iterations = 2)
		erosion = cv2.erode(dilation,kernel,iterations = 3)
		dilation = cv2.dilate(erosion,kernel,iterations = 15)
		mask_closed=cv2.morphologyEx(dilation,cv2.MORPH_CLOSE,kernel)
		mask_cleaned=cv2.morphologyEx(mask_closed,cv2.MORPH_OPEN,kernel)

		# kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
		# image = cv2.erode(image,kernel,iterations = 3)

		return mask_cleaned

	def apply_flood_filling(mask):
		# Threshold.
		# Set values equal to or above 220 to 0.
		# Set values below 220 to 255.
		im_in = mask.copy()
		th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY);
		 
		# Copy the thresholded image.
		im_floodfill = im_th.copy()
		 
		# Mask used to flood filling.
		# Notice the size needs to be 2 pixels than the image.
		h, w = im_th.shape[:2]
		mask = np.zeros((h+2, w+2), np.uint8)
		 
		# Floodfill from point (0, 0)
		cv2.floodFill(im_floodfill, mask, (0,0), 255);
		 
		# Invert floodfilled image
		im_floodfill_inv = cv2.bitwise_not(im_floodfill)
		 
		# Combine the two images to get the foreground.
		im_out = im_th | im_floodfill_inv

		return im_out

	

	#Apply some Pre-processing on the image
	image,image_blur,image_blur_hsv = preprocessing(image)

	#Get the appropriate colour masks
	color_mask = get_color_mask(image_blur_hsv)

	#Apply morphological transformations to the mask
	color_morphed_mask = apply_morphology(color_mask)

	#Apply flood filling on the mask
	color_filled_mask = apply_flood_filling(color_morphed_mask)

	#Use the color_mask on the image
	colour_masked_image = cv2.bitwise_and(image,image,mask = color_filled_mask)

	# display(colour_masked_image,"colour_masked_image")

	#Obtain contours of the colour_masked image and get avg contour area
	val = get_contours(colour_masked_image)


	if(val==0):
		return 0
	fruit_image,fruit_contour,fruit_contour_params = val
	
	
	return val

def get_distance_to_camera(image,knownWidth, perWidth):
	#Obtain focalLength from training or direct specs
	focalLength = 1200
	# compute and return the distance from the maker to the camera
	distance = (knownWidth * focalLength) / perWidth
	# cv2.putText(image, "%.2fcm"%(distance),(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 1)
	return image,distance

def get_width():

	apple = 7.5

	return apple

def get_adjusted_window_width(fruit_window_width,fruit_contour_params):
	#Use the parameters of the tracked_contour to determine the distance to the fruit
	poly_approx_len,area,no_of_contours = fruit_contour_params

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

def get_distance(p1,p2):

	return int(math.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2  ))

def track_fruit(fruit_image,fruit_window,fruit_contour_params,cap):
	
	#Real Width of fruit
	fruit_width = get_width()

	print("Fruit window:\n","[x, y, w, h]:",fruit_window)


	fruit_window_width = fruit_window[2]

	#Calculate window width ,window:(x,y,w,h) and adjust window width based on params
	fruit_window_width = get_adjusted_window_width(fruit_window_width,fruit_contour_params)
	
	#Based on number of contours, get the distance to the object by tweaking parameters
	fruit_image, distance = get_distance_to_camera(fruit_image,fruit_width,fruit_window_width)
	global pivot_x
	global pivot_y
	pivot_y,pivot_x = int(fruit_image.shape[0]/2),int(fruit_image.shape[1]/2)
	cv2.circle(fruit_image,(pivot_x,pivot_y),5,[255,255,0],-1)

	# display(fruit_image,"fruit_image")
	global pd_1
	global pd_2
	global pd_3

	if(pd_1==0):
		pd_1 = distance

	elif(not(pd_1==0) and pd_2==0):
		pd_2 = pd_1
		pd_1 = distance

	elif(not(pd_1==0) and not(pd_2==0)):
		pd_3 = pd_2
		pd_2 = pd_1
		pd_1 = distance
	

	print("Distance:",int(distance))
	

	# if(not(cap==1)):
	# 	move_drone(1)
	# 	return
	
	#Check for consistency
	if( not(pd_1==0) and not(pd_1==0) and not(pd_2==0) and not(pd_3==0) and (abs(pd_1-pd_2)<10 or abs(pd_2-pd_3)<10 or abs(pd_3-pd_1)<10)):
		print("TRUE")
		#Case 1: Far Distance
		if(distance>160 or fruit_contour_params[2]>10):
			#Get circles using moments or Hough Transforms when image is close and apply mask
			# fruit_image = apply_Hough_Transform(fruit_image)
			val=get_fruit(frame,variance)
			fruit_window = get_window(val)
			direction  = get_direction(fruit_window,distance)
			cv2.circle(fruit_image,(int(direction[-1][0]),int(direction[-1][1])),5,[0,255,0],-1)
			print(direction,"DIR")
			move_drone(fruit_image, direction[:-1],"Far")



		#Case 2: Close Distance
		elif(distance<=100):
			#Use canny or sobel edge detection if the fruit is close to the camera
			# ret, frame = cap.read()
			bilateral_filtered_image = cv2.bilateralFilter(fruit_image, 5, 175, 175)
			edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)

			#Find the stack using the edge-detected image
			find_stalk(edge_detected_image)
			direction  = get_direction(fruit_window,distance)
			cv2.circle(fruit_image,(int(direction[-1][0]),int(direction[-1][1])),5,[0,255,0],-1)
			print(direction,"DIR")
			move_drone(fruit_image, direction[:-1],"Close")			


		#Case 3: Medium Distance
		else:
			#Use camshift algorithm to track the fruit window 
			# direction = camshift(fruit_image,fruit_window,cap)
			direction  = get_direction(fruit_window,distance)
			cv2.circle(fruit_image,(int(direction[-1][0]),int(direction[-1][1])),5,[0,255,0],-1)
			print(direction,"DIR")
			#Send commands to the drone to go close to it in the direction
			move_drone(fruit_image,direction[:-1],"Medium")


	else:
		print("FALSE",pd_1,pd_2,pd_3)

	print()


if __name__ == '__main__':
	pd_1 = 0
	pd_2 = 0
	pd_3 = 0
	pivot_x,pivot_y = 0,0
	img_name = sys.argv[1]
	x=0
	#If input is a Video
	if(len(sys.argv)>2):
		
		x=1
		cap = cv2.VideoCapture(img_name)
		for i in range(5):
			cap.read()
		if(cap.isOpened()):
			ret, frame = cap.read()
			
		while(cap.isOpened()):
			ret, frame = cap.read()

			if not ret:
				break
			variance = cv2.Laplacian(frame, cv2.CV_64F).var()
			#Later check variance according to contour size
			
			val=get_fruit(frame,variance)
			if(val==0):
				continue
			fruit_window = get_window(val)
			fruit_image,fruit_contour,fruit_contour_params = val
			track_fruit(fruit_image,fruit_window,fruit_contour_params,cap)


	#If input is an Image
	else:
		
		x=0
		img = cv2.imread(img_name)
		variance = cv2.Laplacian(img, cv2.CV_64F).var()

		val = get_fruit(img,variance)
		fruit_image,fruit_contour,fruit_contour_params = val

		fruit_window_width = get_window_width(val)

		track_fruit(fruit_image,fruit_window_width,fruit_contour_params,1)
