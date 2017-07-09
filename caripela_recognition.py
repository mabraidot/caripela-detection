import sys, os
from time import sleep
import cv2
import numpy as np
from FPS.WebcamVideoStream import WebcamVideoStream


from os import environ
env = dict(environ)
env['LD_PRELOAD'] = '/usr/lib/arm-linux-gnueabihf/libv4l/v4l2convert.so'


face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
model = cv2.createLBPHFaceRecognizer()
model.load('conocidos.xml')
names = {}
with open('conocidos.csv','r') as f:
	for line in f:
		label,name = line.split(',')
		names[int(label)] = str(name).rstrip('\n').strip().strip('"')

#print names

cantidad_fotos = 10
intervalo = 1000 # Intervalo de tiempo para tomar la foto
fotos_tomadas = 0

def knownFace(image):
	w,h = image.shape
	if not image is None and (w > 0 and h > 0):
		prediction = model.predict(image)
		print prediction
		if prediction[1] < 100 and names[int(prediction[0])]:
			return '%s - %s' % (names[int(prediction[0])], str(prediction[1]))
	return False


def extract_features(image):

	global fotos_tomadas
	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags = cv2.CASCADE_SCALE_IMAGE
	)

	for (x, y, w, h) in faces:
		# Solo tomaremos las caras reconocidas que cerca de la camara
		if h > 300:
			knownName = knownFace(gray[y-25:y+h+25, x-25:x+w+25])
			if not knownName:
				if fotos_tomadas < cantidad_fotos:
					cv2.imwrite('Caras/cara_desconocida_'+str(fotos_tomadas)+'.jpg', gray[y-25:y+h+25, x-25:x+w+25])
				
				text = str("Desconocido")
				fotos_tomadas += 1
			else:
				text = str(knownName)
			
			cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
			wText, h = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
			cv2.rectangle(image, (x, y), (x+wText[0]+10, y+20), (0, 255, 0), cv2.cv.CV_FILLED)
			cv2.putText(image, text, (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),2)

	cv2.imshow('img', image)



def inicio():
	opcion = raw_input('\nParese frente a la camara y presione (r: reconococer rostro): ')
	if opcion == 'r':
		return
	else:
		inicio()
	return

inicio()

cam = WebcamVideoStream(src=-1).start()
sleep(1)
print '\nAl finalizar presione (q: Salir del programa).'
while cam.stream.isOpened():
	frame = cam.read()
	extract_features(frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cam.stop()
cv2.destroyAllWindows()

