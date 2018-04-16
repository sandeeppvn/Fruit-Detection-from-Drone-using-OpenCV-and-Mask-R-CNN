#Libraries to import
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
	image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	image, contours, hierarchy = cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(image, contours, -1,(125,0,125), 1)

	contour_list = []
	for contour in contours:
		approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
		area = cv2.contourArea(contour)		
		contour_list.append([contour,len(approx),area])
	
	contour_list.sort(key=lambda contour: contour[2],reverse=True)
	
	track_contour = []
	for i,(contour,approx,area) in enumerate(contours):
		#Some condition to get the right apple wrt to len(contours), approx and area
		if(i==int(len(contours)/4)):
			cv2.drawContours(image, [contour], 0,(255,255,0), 3)
			track_contour = [contour,len(approx),area]
			break

	return image,track_contour

def apply_morphology(mask):
	# kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
	kernel = np.ones((3,3),np.uint8)
	dilation = cv2.dilate(mask,kernel,iterations = 8)
	erosion = cv2.erode(dilation,kernel,iterations = 3)
	dilation = cv2.dilate(erosion,kernel,iterations = 7)
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
	return contour

def display(image,msg):
	maxsize=max(image.shape)
	scale=700/maxsize
	image=cv2.resize(image,None,fx=scale,fy=scale)
	cv2.imshow(msg,image)
	cv2.waitKey(0)

def find_stalk(image):

	pass

def camshift(window):

	pass

def get_fruit(image,variance):

	#Apply some Pre-processing on the image
	image,image_blur,image_blur_hsv = preprocessing(image)

	#Get the appropriate colour masks
	color_mask = get_color_mask(image_blur_hsv)

	#Apply morphological transformations to the mask
	color_morphed_mask = apply_morphology(color_mask)

	#Use the color_mask on the image
	colour_masked_image = cv2.bitwise_and(image,image,mask = color_morphed_mask)

	#Obtain contours of the colour_masked image and get avg contour area
	contour_image, (tracked_contour, poly_approx_len, area) = get_contours(colour_masked_image)

	#Get the rectangular window by approximating tracked_contour
	window = make_window(tracked_contour)

	return contour_image,window,[no_of_contours,poly_approx_len,area]

def move_drone(direction):

	pass
	
def track_fruit(fruit_image,fruit_window,params):

	#Use the parameters of the tracked_contour to determine the distance to the fruit
	no_of_contours,poly_approx_len,area = params

	#Set the threshold parameters, x1 -> Partially close, apply HoughTranforms to detect circles

	x1_contours_len,x1_len,x1_area = (0,0,0)
	x2_contours_len,x2_len,x2_area = (0,0,0)

	#Case 1: Medium Distance
	if((poly_approx_len > x1_len) and (area> x1_area) and (no_of_contours>x1_contours_len)):
		#Get circles using moments or Hough Transforms when image is close and apply mask
		fruit_image = apply_Hough_Transform(fruit_image)

	#Case 2: Close Distance
	if((poly_approx_len > x2_len) and (area> x2_area) and (no_of_contours>x2_contours_len)):
		#Use canny or sobel edge detection if the fruit is close to the camera
		bilateral_filtered_image = cv2.bilateralFilter(colour_masked_image, 5, 175, 175)
		edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)

		#Find the stack using the edge-detected image
		find_stalk(edge_detected_image)

	#Case 3: Far Distance
	else:
		#Use camshift algorithm to track the fruit window 
		direction = camshift(window)

		#Send commands to the drone to go close to it in the direction
		move_drone(direction)



	#Display
	#display(colour_masked_image,"masked image")
	display(contour_image,"contour_image")
	# display(edge_detected_image,"edge image")

if __name__ == '__main__':
	img_name = sys.argv[1]
	
	#If input is a Video
	if(len(sys.argv)>2):
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
		img = cv2.imread(img_name)
		variance = cv2.Laplacian(img, cv2.CV_64F).var()
		fruit_image,fruit_window,fruit_params=get_fruit(img,variance)
		track_fruit(fruit_image,fruit_window,fruit_params)
