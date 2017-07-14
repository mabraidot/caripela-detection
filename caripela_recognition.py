# -*- coding: utf-8 -*-
import sys, os
from time import sleep
import cv2
import numpy as np
from FPS.WebcamVideoStream import WebcamVideoStream

# En Linux parece haber un problema con libv4l y se necesita recargarla
from os import environ
env = dict(environ)
env['LD_PRELOAD'] = '/usr/lib/arm-linux-gnueabihf/libv4l/v4l2convert.so'


###-------------------------------------------------###
### 		CONFIGURACIONES 		    ###
###-------------------------------------------------###
windows = False;		# Está corriendo en windows?
cantidad_fotos = 20		# Cantidad de fotos que se le tomarán a los desconocidos
intervalo = 10 			# Intervalo de tiempo para tomar cada foto (frames por segundo)
fotos_tomadas = 0		# Contador de fotos capturadas
margen_marco = 25		# Cantidad de píxeles que achicaremos las fotos capturadas
tiempo_transcurrido = 0		# Contador de tiempo
humbral_reconocimiento = 55	# Sensibilidad de reconocimiento, menos es más sensible
###-------------------------------------------------###



# Si es windows, la cámara usb está en el índice 0
if windows:
	camIndex = 0
else: 
	camIndex = -1


# Cargamos el xml entrenado para reconocer caras genéricas
face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
# Si existe, cargamos nuestro xml entrenado para identificar caras
modelo = cv2.createLBPHFaceRecognizer()
if os.path.exists('conocidos.xml'):
	modelo.load('conocidos.xml')

# Inicializamos un arreglo con los nombres de las personas conocidas
nombres = {}
with open('conocidos.csv','r') as f:
	for renglon in f:
		id,nombre = renglon.split(',')
		nombres[int(id)] = str(nombre).rstrip('\n').strip().strip('"')
#print nombres


# Función que recibe una imágen con un rostro capturado, Intenta identificarlo 
# con datos del entrenamiento y devuelve su nombre si tiene éxito o False si no.
def esUnaCaraConocida(imagen):
	global humbral_reconocimiento
	
	# Si la imágen existe, intentamos identificarla usando el algoritmo 
	# predict que tiene opencv, si la predicción está dentro del humbral, 
	# la damos por buena y retornamos el nombre de la persona identificada.
	w,h = imagen.shape
	if not imagen is None and (w > 0 and h > 0) and len(nombres) > 0:
		prediccion = modelo.predict(imagen)
		if prediccion[1] < humbral_reconocimiento and nombres[int(prediccion[0])]:
			return '%s - %s' % (nombres[int(prediccion[0])], str(prediccion[1]))
	return False



def extract_features(image):

	global fotos_tomadas, tiempo_transcurrido, intervalo, cantidad_fotos, margen_marco
	
	gray = image.copy()
	gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
	#gray = cv2.equalizeHist(gray)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	gray = clahe.apply(gray)

	faces = face_cascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags = cv2.CASCADE_SCALE_IMAGE
	)

	for (x, y, w, h) in faces:
		# Solo tomaremos las caras reconocidas que cerca de la camara
		if h > 250:
			#knownName = esUnaCaraConocida(gray[y-25:y+h+25, x-25:x+w+25])
			knownName = esUnaCaraConocida(gray[y+margen_marco:y+h-margen_marco, x+margen_marco:x+w-margen_marco])
			if not knownName:
				if (tiempo_transcurrido % intervalo) == 0 and fotos_tomadas < cantidad_fotos:
					#cv2.imwrite('Caras/cara_desconocida_'+str(fotos_tomadas)+'.jpg', gray[y-25:y+h+25, x-25:x+w+25])
					cv2.imwrite('Caras/cara_desconocida_'+str(fotos_tomadas)+'.jpg', 
							gray[y+margen_marco:y+h-margen_marco, 
							x+margen_marco:x+w-margen_marco])
					fotos_tomadas += 1
				text = str("Desconocido - Foto "+str(fotos_tomadas))
				tiempo_transcurrido += 1
			else:
				text = str(knownName)
			
			cv2.rectangle(image, (x+margen_marco, y+margen_marco), (x+w-margen_marco, y+h-margen_marco), (0, 255, 0), 2)
			wText, h = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
			cv2.rectangle(image, (x+margen_marco, y+margen_marco), (x+margen_marco+wText[0]+10, y+margen_marco+20), (0, 255, 0), cv2.cv.CV_FILLED)
			cv2.putText(image, text, (x+margen_marco+5, y+margen_marco+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),2)

	cv2.imshow('img', image)
	return



def inicio():
	opcion = raw_input('\nParese frente a la camara y presione (r: reconococer rostro), \nAl finalizar presione (q: Salir del programa):')
	if opcion == 'r':
		return
	else:
		inicio()
	return

inicio()

cam = WebcamVideoStream(src=camIndex).start()
sleep(1)
print '\nAl finalizar presione (q: Salir del programa).'
while cam.stream.isOpened():
	frame = cam.read()
	extract_features(frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cam.stop()
cv2.destroyAllWindows()
