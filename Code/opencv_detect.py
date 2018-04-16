import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

def convex_hull(contour):
	# for each contour
	for cnt in contours:
	    # get convex hull
		if(cnt.all() == biggest_contour.all()):
		    hull = cv2.convexHull(cnt)
		    # draw it in red color
		    cv2.drawContours(mask, [hull], -1, (0, 0, 255), 1)

def find_contours(image):
	contour_image=image.copy()

	# bilateral_filtered_image = cv2.bilateralFilter(contour_image, 5, 175, 175)
	# edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)
	# cv2.imshow('Edge', edge_detected_image)
	# cv2.waitKey(0)

	# circles = cv2.HoughCircles(image,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
	# circles = np.uint16(np.around(circles))
	# mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)  # mask is only
	# for i in circles[0, :]:
	# 	cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), -1)
	# # get first masked value (foreground)
	# fg = cv2.bitwise_or(img, img, mask=mask)
	#
	# # get second masked value (background) mask must be inverted
	# mask = cv2.bitwise_not(mask)
	# background = np.full(img.shape, 255, dtype=np.uint8)
	# bk = cv2.bitwise_or(background, background, mask=mask)
	#
	# # combine foreground+background
	# image = cv2.bitwise_or(fg, bk)
	#
	# hull = cv2.convexHull(biggest_contour)
	# defects = cv2.convexityDefects(biggest_contour,hull)

	_ , contours , hierarchy=cv2.findContours(contour_image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	contour_sizes=[(cv2.contourArea(contour),contour) for contour in contours]

	#If no contours, return 0
	if(len(contour_sizes)<=0):
		return 0

	#If contour area is between a threshold, then use it
	fruit_contours = list()
	for index,i in enumerate(contour_sizes):
		# if(i[0]>2500 and i[0]<3500):
		if(True):
			print("Area:",i[0])
			# cv2.drawContours(contour_image, [i[1]], -1, (115,110,255), 6)
			ellipse = cv2.fitEllipse(cnt)
			cv2.ellipse(img,ellipse,(0,255,0),2)
			fruit_contours.append(i[1])
			print(i[1].shape)

	#Create the mask with apple_contour
	# mask=np.zeros(image.shape,np.uint8)
	#TO-DO


	return fruit_contours,contour_image

def get_mask(image_blur_hsv):
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



	mask1=cv2.inRange(image_blur_hsv,min_color,max_color)
	mask2=cv2.inRange(image_blur_hsv,min_color2,max_color2)

	mask=mask1+mask2
	return mask

def get_ellipse():
	kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
	# kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30))
	# kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(85,85))
	# kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

	return kernel

def draw_fruit(image):

	#PRE PROCESSING OF IMAGE
	# image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	maxsize=max(image.shape)
	scale=700/maxsize
	image=cv2.resize(image,None,fx=scale,fy=scale)
	image_blur=cv2.GaussianBlur(image,(7,7),0)
	image_blur_hsv=cv2.cvtColor(image_blur,cv2.COLOR_RGB2HSV)

	bilateral_filtered_image = cv2.bilateralFilter(image_blur_hsv, 5, 175, 175)
	edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)

	edge_detected_image = image_blur.copy()

	# kernel = get_ellipse()
	kernel = np.ones((5,5),np.uint8)
	#Dilate and erode to remove blobs/noises
	dilation = cv2.dilate(edge_detected_image,kernel,iterations = 2)
	erosion = cv2.erode(edge_detected_image,kernel,iterations = 3)
	dilation = cv2.dilate(edge_detected_image,kernel,iterations = 5)
	mask_closed=cv2.morphologyEx(edge_detected_image,cv2.MORPH_CLOSE,kernel)
	mask_cleaned=cv2.morphologyEx(edge_detected_image,cv2.MORPH_OPEN,kernel)

	cv2.imshow('Edge', edge_detected_image)
	cv2.waitKey(0)

	mask = get_mask(image_blur_hsv)


	#Find the recognized fruit contours on the mask
	temp=find_contours(mask_cleaned)

	#If no contours are found, return 0
	if(temp==0):
		return 0

	big_contour,contour_image = temp
	# cv2.imshow("cnt",contour_image)
	# cv2.waitKey(0)

	# rgb_mask=cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
	# img=cv2.addWeighted(rgb_mask,0.5,image,0.5,0)
	# return img


	#Draw the masks on the image
	overlay_image = cv2.bitwise_and(contour_image,contour_image,mask = mask)
	blur = cv2.GaussianBlur(overlay_image,(7,7),0)
	ret,im_th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	image=cv2.cvtColor(overlay_image,cv2.COLOR_GRAY2BGR)
	if(image.shape[0]>0 and image.shape[1]>0):
		cv2.imshow("img",res)
		cv2.waitKey(0)
	# 	cv2.destroyAllWindows()
	return image

if __name__ == '__main__':
	img_name = sys.argv[1]

	#If input is a Video
	if(len(sys.argv)>2):
		print("here")
		cap = cv2.VideoCapture(img_name)
		cap.read()
		cap.read()
		cap.read()
		cap.read()
		while(cap.isOpened()):

			ret, frame = cap.read()
			if not ret:
				break
			i = cv2.Laplacian(frame, cv2.CV_64F).var()
			#Check for blur images, If not blurry

			if(i<4):
				continue
			print("Laplacian Viariance:",i)
			res = cv2.resize(frame,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)

			result=draw_fruit(frame)

			cv2.destroyAllWindows()

	#If input is an image
	else:
		img = cv2.imread(img_name)
		result=draw_fruit(img)
