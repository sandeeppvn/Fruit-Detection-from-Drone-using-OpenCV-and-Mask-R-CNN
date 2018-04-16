import cv2
import sys
import numpy as np

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

def get_contours(image):
	# kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
	# image = cv2.erode(image,kernel,iterations = 3)
	im = image.copy()
	image=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	image, contours, hierarchy = cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	# cv2.drawContours(im, contours, -1,(255,255,0), 2)

	contour_list = []
	for contour in contours:
		approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
		area = cv2.contourArea(contour)		
		contour_list.append([contour,len(approx),area])
	
	contour_list.sort(key=lambda contour: contour[2],reverse=True)
	
	track_contour = []
	#Get the largest contour
	track_contour = contour_list[0]

	# for i,(contour,approx,area) in enumerate(contour_list):
	# 	#Some condition to get the right apple wrt to len(contours), approx and area
	# 	if(i==int(len(contours)/4)):
	# 		# cv2.drawContours(image, [contour], 0,(255,255,0), 3)
	# 		track_contour = [contour,approx,area]
	# 		break

	return im,track_contour,len(contours)

def apply_morphology(mask):
	# kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
	kernel = np.ones((3,3),np.uint8)
	dilation = cv2.dilate(mask,kernel,iterations = 2)
	erosion = cv2.erode(dilation,kernel,iterations = 3)
	dilation = cv2.dilate(erosion,kernel,iterations = 15)
	mask_closed=cv2.morphologyEx(dilation,cv2.MORPH_CLOSE,kernel)
	mask_cleaned=cv2.morphologyEx(mask_closed,cv2.MORPH_OPEN,kernel)

	return mask_cleaned

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

def make_window(contour):
	#Make a rectangular window for the contour to track
	(x,y,w,h) = cv2.boundingRect(contour)
	return (x,y,w,h)

def display(image,msg):
	maxsize=max(image.shape)
	scale=700/maxsize
	image=cv2.resize(image,None,fx=scale,fy=scale)
	cv2.imshow(msg,image)
	cv2.waitKey(x)

def find_stalk(image):

	pass

def camshift(window):
	track_window = (x,y,w,h)
	# set up the ROI for tracking
	roi = frame[x:x+h, y:y+w]
	hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
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
	        cv2.imshow('img2',img2)
	        k = cv2.waitKey(60) & 0xff
	        if k == 27:
	            break
	        else:
	            cv2.imwrite(chr(k)+".jpg",img2)
	    else:
	        break

def move_drone(direction):

	pass

def apply_convexHull(img,contour):
	hull = cv2.convexHull(contour)
	cv2.drawContours(img, [hull], 0, [0, 255, 0])
	display(img,"a")

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

def get_fruit(image,variance):

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

	#Obtain contours of the colour_masked image and get avg contour area
	val = get_contours(colour_masked_image)
	print(len(val))
	if(len(val)==0):
		return
	contour_image, (tracked_contour, poly_approx_len, area),no_of_contours = val

	#Use convex hull on the traked mask
	apply_convexHull(contour_image, tracked_contour)

	#Get the rectangular window by approximating tracked_contour
	window = make_window(tracked_contour)

	#Display
	#display(colour_masked_image,"masked image")
	display(contour_image,"contour_image")
	# display(edge_detected_image,"edge image")

	return contour_image,window,[no_of_contours,poly_approx_len,area]

def track_fruit(fruit_image,fruit_window,params):

	#Use the parameters of the tracked_contour to determine the distance to the fruit
	no_of_contours,poly_approx_len,area = params

	#Set the threshold parameters, x1 -> Partially close, apply HoughTranforms to detect circles

	x1_contours_len,x1_len,x1_area = (11111110,11111110,11111110)
	x2_contours_len,x2_len,x2_area = (11111110,11111110,11111110)

	#Case 1: Medium Distance
	if((poly_approx_len > x1_len) and (area> x1_area) and (no_of_contours>x1_contours_len)):
		#Get circles using moments or Hough Transforms when image is close and apply mask
		fruit_image = apply_Hough_Transform(fruit_image)

	#Case 2: Close Distance
	if((poly_approx_len > x2_len) and (area> x2_area) and (no_of_contours>x2_contours_len)):
		#Use canny or sobel edge detection if the fruit is close to the camera
		bilateral_filtered_image = cv2.bilateralFilter(fruit_image, 5, 175, 175)
		edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)

		#Find the stack using the edge-detected image
		find_stalk(edge_detected_image)

	#Case 3: Far Distance
	else:
		#Use camshift algorithm to track the fruit window 
		direction = camshift(fruit_window)

		#Send commands to the drone to go close to it in the direction
		move_drone(direction)


if __name__ == '__main__':
	img_name = sys.argv[1]
	x=0
	#If input is a Video
	if(len(sys.argv)>2):
		
		x=1
		cap = cv2.VideoCapture(img_name)
		for i in range(5):
			cap.read()
		while(cap.isOpened()):
			ret, frame = cap.read()
			if not ret:
				break
			variance = cv2.Laplacian(frame, cv2.CV_64F).var()
			#Later check variance according to contour size
			fruit_image,fruit_window,fruit_params=get_fruit(frame,variance)
			track_fruit(fruit_image,fruit_window,fruit_params)


	#If input is an Image
	else:
		
		x=0
		img = cv2.imread(img_name)
		variance = cv2.Laplacian(img, cv2.CV_64F).var()
		fruit_image,fruit_window,fruit_params=get_fruit(img,variance)
		track_fruit(fruit_image,fruit_window,fruit_params)
