import cv2
import numpy as np

img = cv2.imread('../Data/Pics/55.png',0)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))

maxi = max(circles[0,:],key=lambda x:x[2])
# cv2.circle(cimg,(maxi[0],maxi[1]),int(1.3*maxi[2]),(0,255,0),-1)
mask = np.zeros(img.shape, np.uint8)
cv2.circle(mask, (maxi[0],maxi[1]), maxi[-1], 255, -1)
# mask = cv2.bitwise_not(mask)

cv2.bitwise_and(cimg,cimg,mask=mask)
res = cv2.resize(mask,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)

cv2.imshow('detected circles',res)
cv2.waitKey(0)
cv2.destroyAllWindows()
