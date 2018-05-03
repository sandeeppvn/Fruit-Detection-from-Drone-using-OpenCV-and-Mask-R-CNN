import cv2
import sys
import numpy as np
from Distance import get_distance

def preprocessing(image):
	#PRE PROCESSING OF IMAGE

	# image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	maxsize=max(image.shape)
	scale=700/maxsize
	image=cv2.resize(image,None,fx=scale,fy=scale)

	image_blur=cv2.GaussianBlur(image,(7,7),0)
	cv2.imwrite("img_blur.png",image_blur)
	image_blur_hsv=cv2.cvtColor(image_blur,cv2.COLOR_BGR2HSV)
	cv2.imwrite("img_blur_hsv.png",image_blur_hsv)

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
	cv2.imwrite("mask.png",mask)
	return mask

def apply_morphology(mask):
	kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	# kernel = np.ones((3,3),np.uint8)
	dilation = cv2.dilate(mask,kernel,iterations = 2)
	cv2.imwrite("dilation1.png",dilation)
	erosion = cv2.erode(dilation,kernel,iterations = 3)
	cv2.imwrite("erosion.png",erosion)
	dilation = cv2.dilate(erosion,kernel,iterations = 15)
	cv2.imwrite("dilation2.png",dilation)
	mask_closed=cv2.morphologyEx(dilation,cv2.MORPH_CLOSE,kernel)
	cv2.imwrite("mask_closed.png",mask_closed)
	mask_cleaned=cv2.morphologyEx(mask_closed,cv2.MORPH_OPEN,kernel)
	cv2.imwrite("mask_cleaned.png",mask_cleaned)
	return mask_cleaned

def apply_flood_filling(mask):

	im_in = mask.copy()
	th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY);
	 
	# Copy the thresholded image.
	im_floodfill = im_th.copy()
	 
	# Mask used to flood filling.Notice the size needs to be 2 pixels than the image.
	h, w = im_th.shape[:2]
	mask = np.zeros((h+2, w+2), np.uint8)
	 
	# Floodfill from point (0, 0)
	cv2.floodFill(im_floodfill, mask, (0,0), 255);
	 
	# Invert floodfilled image
	im_floodfill_inv = cv2.bitwise_not(im_floodfill)
	 
	# Combine the two images to get the foreground.
	im_out = im_th | im_floodfill_inv

	return im_out

def get_contours(image):
		
		im = image.copy()
		image=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		image, contours, hierarchy = cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		
		contour_list = []
		for contour in contours:
			#Approximate the contour using convexHull
			hull = cv2.convexHull(contour)
			#The length of the approxPolyDP is used to depict the shape of the contour
			approx = cv2.approxPolyDP(hull,0.1*cv2.arcLength(hull,True),True)
			area = cv2.contourArea(hull)
			# if(cv2.isContourConvex(approx)):
			contour_list.append([hull,len(approx),area])

		#Get the largest contour out of all the contours
		contour_list.sort(key=lambda contour: contour[2],reverse=True)
		
		if(len(contour_list)==0):
			return 0

		if(len(contour_list)==1):
			contour,approx,area = contour_list[0]
		else:
			contour,approx,area = contour_list[1]

		tracked_contour_params = [approx,area,len(contours)]
		

		return im,contour,tracked_contour_params

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
	cv2.imwrite("colour_masked_image.png",colour_masked_image)

	#Obtain contours of the colour_masked image and get avg contour area
	contour_val = get_contours(colour_masked_image)

	if(contour_val==0):
		return 0

	# im,contour,tracked_contour_params = contour_val
	return contour_val

def get_window(contour_val):

	if(isinstance(contour_val, int)):
		return 0

	#Make a rectangular window for the contour to track
	tracked_contour_image,tracked_contour,(poly_approx_len,area,no_of_contours) = contour_val
	
	# draw a bounding rotated box around the image and display it
	rect = cv2.minAreaRect(tracked_contour)
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	cv2.drawContours(tracked_contour_image,[box],0,(0,0,255),2)
	cv2.imwrite("window.png",tracked_contour_image)
	window_width = ( get_distance(box[0],box[1]) + get_distance(box[2],box[3]) )/2.0
	window_height = ( get_distance(box[0],box[3]) + get_distance(box[1],box[2]) )/2.0

	mid_x = (box[0][0]+box[1][0]+box[2][0]+box[3][0])/4
	mid_y = (box[0][1]+box[1][1]+box[2][1]+box[3][1])/4
	window = [mid_x,mid_y,window_width,window_height]

	return window

def extract_fruit(image,variance):
	contour_val = get_fruit(image,variance)
	window = get_window(contour_val)
	if(isinstance(window, int)):
		return 0
	img,contour,contour_params = contour_val

	return contour_val,window

