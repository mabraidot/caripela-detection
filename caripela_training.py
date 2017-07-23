# -*- coding: utf-8 -*-
import sys, os
from time import sleep
import cv2
import numpy as np
from FPS.WebcamVideoStream import WebcamVideoStream


from os import environ
env = dict(environ)
env['LD_PRELOAD'] = '/usr/lib/arm-linux-gnueabihf/libv4l/v4l2convert.so'

#face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
#recognizer = cv2.createLBPHFaceRecognizer()


def entrenar():
	
	(imagenes, etiquetas, nombres, id) = ([], [], {}, 0)
	with open('conocidos.csv','r') as csv_nombres:
		for renglon in csv_nombres:
			etiqueta,nombre = renglon.split(',')
			#nombres[int(etiqueta)] = str(nombre).rstrip('\n').strip().strip('"')
			id = int(etiqueta)+1

	
	nombre = raw_input('\nIngrese el nombre de la persona y presione <Enter>: ')	
	BASE_PATH = "Caras/"
	SEPARATOR=";"
	nombres[id] = nombre
	etiqueta = id
	
	# Creamos el directorio donde se moveran las imágenes, 
	# el nombre del directorio es el id de la persona
	"""directorio = BASE_PATH+"/"+str(id)
	if not os.path.exists(directorio):
		try:
			os.makedirs(directorio)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise
	"""	
	
	for carpeta, subcarpetas, fotos in os.walk(BASE_PATH):
		#for foto in os.listdir(carpeta):
		for foto in fotos:	
			ruta = os.path.join(carpeta, foto)
			#print "%s%s%d" % (ruta, SEPARATOR, etiqueta)
			if os.path.exists(ruta):
				imagenes.append(cv2.imread(ruta, 0))
				etiquetas.append(int(etiqueta))
				#os.rename(ruta, directorio+"/"+foto)
				os.remove(ruta)
				
	(imagenes, etiquetas) = [np.array(lista) for lista in [imagenes, etiquetas]]
	if len(imagenes) > 0:
		modelo = cv2.createLBPHFaceRecognizer()
		archivo_modelo = 'conocidos.xml'
		if os.path.exists(archivo_modelo):
			modelo.load(archivo_modelo)
			modelo.update(imagenes, etiquetas)
			print 'Actualizando el modelo...'
		else:
			modelo.train(imagenes, etiquetas)
			print 'Creando un nuevo modelo...'
		modelo.save(archivo_modelo)
		csv_nombres = open('conocidos.csv', 'a+')
		csv_nombres.write(str(id)+', "'+nombre+'"\n')
		csv_nombres.close()
		
		# Luego de entrenar al algoritmo, eliminamos las imágenes
		for carpeta, subcarpetas, fotos in os.walk(BASE_PATH):
			for foto in fotos:	
				ruta = os.path.join(carpeta, foto)
				os.remove(ruta)
				
	else:
		print u"No hay imágenes disponibles para realizar el entrenamiento."
	exit()


"""def inicio():
	opcion = raw_input('\nPresione (t: entrenar): ')
	if opcion == 't':
		entrenar()
	else:
		inicio()
	return


inicio()
"""
entrenar()

