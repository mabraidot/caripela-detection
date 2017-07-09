import sys
from time import sleep
import cv2

from os import environ
env = dict(environ)
env['LD_PRELOAD'] = '/usr/lib/arm-linux-gnueabihf/libv4l/v4l2convert.so'

windows = True;
if windows:
	camIndex = 0
else: 
	camIndex = -1

if windows:
	face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye.xml')
else:
	face_cascade = cv2.CascadeClassifier('/home/ifts14-6/Descargas/haar/haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('/home/ifts14-6/Descargas/haar/haarcascade_eye.xml')


def extract_features(image, gray):
	
	#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

		#noses = nose_cascade.detectMultiScale(roi_gray, minSize=(100, 30))
		#for (ex,ey,ew,eh) in noses:
		#	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)


	cv2.imshow('img', image)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()


def tomarFoto():
	cam = cv2.VideoCapture(-1)
	ret, imagen = cam.read()
	cv2.waitKey(10)
	cam.release()
	extract_features(imagen)

#while 1:
#	raw_input("Presione Enter para continuar...")
#	tomarFoto()


cap = cv2.VideoCapture(camIndex)
cap.set(3 , 1024);
cap.set(4 , 960);
sleep(1)

while cap.isOpened():
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	extract_features(frame,gray)
	#cv2.imshow('img', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	
cap.release()
cv2.destroyAllWindows()

