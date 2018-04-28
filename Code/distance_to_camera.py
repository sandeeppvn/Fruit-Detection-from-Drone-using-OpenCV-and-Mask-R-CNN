# USAGE
# python distance_to_camera.py

# import the necessary packages
import numpy as np
import cv2


def get_window_width(val):
	#Make a rectangular window for the contour to track
	contour_image,track_contour_features ,no_of_contours = val
	poly_approx_len,tracked_contour, area = track_contour_features
	print()
	print("Tracked Contour Fetaures:")
	print("\tNo of cont:",no_of_contours)
	print("\tPoly_approx_len:",poly_approx_len)
	print("\tArea:",area)
	

	# draw a fitting ellipse around the image and display it
	ellipse = cv2.fitEllipse(tracked_contour)
	cv2.ellipse(contour_image,ellipse,(0,255,0),2)


	# draw a bounding rotated box around the image and display it
	rect = cv2.minAreaRect(tracked_contour)
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	cv2.drawContours(contour_image,[box],0,(0,0,255),2)

	# print("Rectangle properties:\n",box)
	display(contour_image,"cnt")

	window_width = abs(box[0][0]-box[-1][0])

	print("Window Width",window_width)

	return window_width
	 

def display(image,msg):
	x = 0
	maxsize=max(image.shape)
	scale=700/maxsize
	image=cv2.resize(image,None,fx=scale,fy=scale)
	cv2.imshow(msg,image)
	cv2.waitKey(x)

def get_fruit(image):

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

	def get_contours(image):
		# kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
		# image = cv2.erode(image,kernel,iterations = 3)
		im = image.copy()
		image=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		image, contours, hierarchy = cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		# cv2.drawContours(im, contours, -1,(255,255,0), 3)
		#Dow list is len(approx),hull,area
		contour_list = [(len(cv2.approxPolyDP(contour,0.1*cv2.arcLength(contour,True),True)),cv2.convexHull(contour),cv2.contourArea(contour)) for contour in contours]
			
		contour_list.sort(key=lambda contour: contour[2],reverse=True)
		track_contour_features = []
		#Get the largest contour
		if(len(contour_list)==1):
			track_contour_features = contour_list[0]

		else:
			track_contour_features = contour_list[1]


		return im,track_contour_features,len(contours)

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
	
	if(len(val)==0):
		print("no contour")
		return

	contour_image,track_contour_features ,no_of_contours = val

	return val

	
def get_distance_to_camera(image,knownWidth, perWidth):
	#Obtain focalLength from training or direct specs
	focalLength = 1
	# compute and return the distance from the maker to the camera
	distance = (knownWidth * focalLength) / perWidth
	cv2.putText(image, "%.2fcm"%(distance),(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,2.0, (0, 255, 0), 3)
	return image,distance

def get_width():

	apple = 7.5

	return apple

def main():

	# initialize the known distance from the camera to the object
	KNOWN_DISTANCE = 200.0

	# initialize the known object width
	KNOWN_WIDTH = 7.5

	# initialize the list of images that we'll be using
	IMAGE_PATHS = ["..\Data\Pics\\55.png", "..\Data\Pics\\125.png","..\Data\Pics\\200.png"]

	img = cv2.imread(IMAGE_PATHS[2])

	val=get_fruit(img)
	window_width = get_window_width(val)

	focalLength = (window_width * KNOWN_DISTANCE) / KNOWN_WIDTH

	print("Focal Length:",focalLength)
	print()


	

main()