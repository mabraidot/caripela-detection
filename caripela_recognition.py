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
###                 CONFIGURACIONES                 ###
###-------------------------------------------------###
windows = False			# Está corriendo en windows?
cantidad_fotos = 20		# Cantidad de fotos que se le tomarán a los desconocidos
intervalo = 10			# Intervalo de tiempo para tomar cada foto (frames por segundo)
fotos_tomadas = 0		# Contador de fotos capturadas
margen_marco = 25		# Cantidad de píxeles que achicaremos las fotos capturadas
tiempo_transcurrido = 0		# Contador de tiempo
humbral_reconocimiento = 45	# Sensibilidad de reconocimiento, menos es más sensible
###-------------------------------------------------###



# Si es windows, la cámara usb está en el índice 0
if windows:
	camIndex = 0
else: 
	camIndex = -1


# Cargamos el xml entrenado para reconocer caras genéricas (Algoritmo Cascadas Haar)
cascada = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
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


"""
Función que recibe una imágen con un rostro capturado, Intenta identificarlo 
con datos del entrenamiento y devuelve su nombre si tiene éxito o False si no.
"""
def esUnaCaraConocida(imagen):
	global humbral_reconocimiento
	
	"""
	Si la imágen existe, intentamos identificarla usando el algoritmo 
	predict que tiene opencv, si la predicción está dentro del humbral, 
	la damos por buena y retornamos el nombre de la persona identificada.
	"""
	w,h = imagen.shape
	if not imagen is None and (w > 0 and h > 0) and len(nombres) > 0:
		prediccion = modelo.predict(imagen)
		if prediccion[1] < humbral_reconocimiento and nombres[int(prediccion[0])]:
			#return '%s - %s' % (nombres[int(prediccion[0])], str(prediccion[1]))
			return '%s' % (nombres[int(prediccion[0])])
	return False


"""
Función que recibe una imágen e intenta detectar rostros en ella.
Si hay rostros, los recuadra e intenta identificarlos usando el algoritmo 
pre-entrenado, si es una cara conocida, imprime el nombre de la persona en el recuadro.
"""
def buscarCaras(imagen):
	
	global fotos_tomadas, tiempo_transcurrido, intervalo, cantidad_fotos, margen_marco
	# Pasamos la imágen a escala de grises, el algoritmo de reconocimiento lo requiere así
	grices = imagen.copy()
	grices = cv2.cvtColor(grices, cv2.COLOR_BGR2GRAY)
	# Ecualizamos la imágen para normalizar la iluminación, ya que el algoritmo de 
	# reconocimiento es muy sensible a variaciones de luces y sombras
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	grices = clahe.apply(grices)
	
	# Buscamos caras en la imágen en escala de grises
	caras = cascada.detectMultiScale(
		grices,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags = cv2.CASCADE_SCALE_IMAGE
	)
	
	# Por cada cara encontrada vamos a dibujar un recuadro y tratar de identificarlas
	for (x, y, w, h) in caras:
		# Solo tomaremos las caras reconocidas que estén cerca de la camara, 
		# cuadro mayor a 250 píxeles
		if h > 250:
			# Buscamos la cara entre nuestras caras previamente identificadas
			nombreConocido = esUnaCaraConocida(grices[y+margen_marco:y+h-margen_marco, x+margen_marco:x+w-margen_marco])
			if not nombreConocido:
				# Si no conocemos la cara, tomamos una cantidad parametrizable de fotos cada cierta
				# cantidad de cuadros y las guardamos en un directorio para realizar el entrenamiento
				if (tiempo_transcurrido % intervalo) == 0 and fotos_tomadas < cantidad_fotos:
					cv2.imwrite('Caras/cara_desconocida_'+str(fotos_tomadas)+'.jpg', 
							grices[y+margen_marco:y+h-margen_marco, 
							x+margen_marco:x+w-margen_marco])
					fotos_tomadas += 1
				texto = str("Desconocido - Foto "+str(fotos_tomadas))
				tiempo_transcurrido += 1
			else:
				# Si es un rostro conocido, tomamos el nombre para mostrar en el recuadro
				texto = str(nombreConocido)
			
			# Dibujamos el recuadro sobre la cara y colocamos la etiqueta con el nombre conocido 
			# o una indicación de que no es un rostro conocido
			cv2.rectangle(imagen, (x+margen_marco, y+margen_marco), (x+w-margen_marco, y+h-margen_marco), (0, 255, 0), 2)
			wText, h = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
			cv2.rectangle(imagen, (x+margen_marco, y+margen_marco), (x+margen_marco+wText[0]+10, y+margen_marco+20), (0, 255, 0), cv2.cv.CV_FILLED)
			cv2.putText(imagen, texto, (x+margen_marco+5, y+margen_marco+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),2)

	# Mostramos cada cuadro en la ventana de video
	cv2.imshow('Video', imagen)

	return


"""
Función que presenta el menú en pantalla para iniciar el reconocimiento
"""
def inicio():
	opcion = raw_input('\nParate frente a la cámara y presioná (r: reconococer rostro), \nAl finalizar presioná (q: Salir del programa):')
	"""
	Si el usuario presionó la letra r, salimos de la función para que el programa 
	continúe y entre en el bucle (while) principal. Si es otra letra, la función
	sigue llamándose a sí misma para que continúe el menú en pantalla
	"""
	if opcion == 'r':
		return
	else:
		inicio()
	return

"""
Lo primero que se debe hacer al iniciar el programa es llamar a inicio() para 
mostrar el menú de opciones
"""
inicio()

# Inicializamos la cámara
camara = WebcamVideoStream(src=camIndex).start()
sleep(1)
# Mientras el programa está corriendo, mostramos en pantalla la opción de salir
print '\nAl finalizar presioná (q: Salir del programa).'

"""
Bucle principal. Si pudimos capturar la cámara, procesamos cada cuadro en 
búsqueda de rostros
"""
while camara.stream.isOpened():
	# Capturamos un cuadro del video
	cuadro = camara.read()
	# Procesamos el cuadro para ver si hay rostros
	buscarCaras(cuadro.copy())
	# Si el usuario presiona la letra de salir, abortamos el bucle
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Al salir, liberar la cámara y cerrar la ventana de video
camara.stop()
cv2.destroyAllWindows()

