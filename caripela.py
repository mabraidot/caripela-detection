import sys
from time import sleep
import cv2
import numpy as np

from os import environ
env = dict(environ)
env['LD_PRELOAD'] = '/usr/lib/arm-linux-gnueabihf/libv4l/v4l2convert.so'

windows = True;
if windows:
	camIndex = 0
else: 
	camIndex = -1


face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye.xml')

def extract_features(image, detectEyes):
	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	faces = face_cascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags = cv2.CASCADE_SCALE_IMAGE
	)

	# iterate over all identified faces and try to find eyes
	for (x, y, w, h) in faces:
		
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = image[y:y+h, x:x+w]
		
		eyes = eye_cascade.detectMultiScale(roi_gray, minSize=(30, 30))
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
			if detectEyes:
				#blur_gray = cv2.medianBlur(roi_gray,2)
				circles = cv2.HoughCircles(
					roi_gray
					,cv2.HOUGH_GRADIENT
					,1
					,20
					,param1      = 60
					,param2      = 25
					,minRadius   = 1
					,maxRadius   = np.round(ew/4).astype("int")
				)
				if circles is not None:
					circles = np.round(circles[0, :]).astype("int")
					for (cx, cy, cr) in circles:
						cv2.circle(roi_color, (cx, cy), cr, (0, 255, 0), 2)
			

		
		#noses = nose_cascade.detectMultiScale(roi_gray, minSize=(100, 30))
		#for (ex,ey,ew,eh) in noses:
		#	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)

	cv2.imshow('img', image)



cap = cv2.VideoCapture(camIndex)
cap.set(3 , 960);
cap.set(4 , 720);
sleep(1)

while cap.isOpened():
	ret, frame = cap.read()
	extract_features(frame,True)
	#cv2.imshow('img', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

