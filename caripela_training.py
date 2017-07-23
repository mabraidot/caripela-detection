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


def menu():
	
	global opciones
	
	opciones = {}
	(nombres, id) = ({}, 0)
	with open('conocidos.csv','r') as csv_nombres:
		for renglon in csv_nombres:
			etiqueta,nombre = renglon.split(',')
			nombre = str.replace(nombre, "\"", "")
			nombre = str.strip(nombre)
			opciones[int(etiqueta)] = nombre
			
	opciones['n'] = "Nuevo rostro"
	
	texto_opciones = "\n\nSi deseás actualizar el modelo de una persona que ya existía, seleccioná la opción con su nombre.\nSi es una persona desconocida, presioná n para entrenar el modelo: \n\n"
	
	for id, texto in opciones.iteritems():
		texto_opciones = texto_opciones + "(" + str(id) + ") " + texto + "\n"
		
	opcion = raw_input(texto_opciones)
	if opcion == 'n':
		nombre = raw_input('\nIngresá el nombre de la persona y presioná <Enter>: ')
		id = int(etiqueta) + 1
		return id, nombre
	elif int(opcion) in opciones:
		nombre = opciones[int(opcion)]
		id = int(opcion)
		return id, nombre
	else:
		print u"Opción no válida."
		exit()
		
		

def entrenar():
	
	global opciones
	
	(imagenes, etiquetas, nombres, id) = ([], [], {}, 0)
	BASE_PATH = "Caras/"
	SEPARATOR=";"
	
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
				etiquetas.append(int(id))
				#os.rename(ruta, directorio+"/"+foto)
				os.remove(ruta)
				
	(imagenes, etiquetas) = [np.array(lista) for lista in [imagenes, etiquetas]]
	if len(imagenes) > 0:
	
	
		(id, nombre) = menu()
		
	
		modelo = cv2.createLBPHFaceRecognizer()
		archivo_modelo = 'conocidos.xml'
		if os.path.exists(archivo_modelo):
			modelo.load(archivo_modelo)
			modelo.update(imagenes, etiquetas)
			print u'Actualizando el modelo...'
		else:
			modelo.train(imagenes, etiquetas)
			print u'Creando un nuevo modelo...'
		modelo.save(archivo_modelo)
		if int(id) not in opciones:
			csv_nombres = open('conocidos.csv', 'a+')
			csv_nombres.write(str(id)+', "'+nombre+'"\n')
			csv_nombres.close()
			print u'Entrenando una nueva cara...'
		else:
			print u'Actualizando una cara existente...'
		
		# Luego de entrenar al algoritmo, eliminamos las imágenes
		for carpeta, subcarpetas, fotos in os.walk(BASE_PATH):
			for foto in fotos:	
				ruta = os.path.join(carpeta, foto)
				os.remove(ruta)
				
	else:
		print u"\n\nNo hay imágenes disponibles para realizar el entrenamiento.\n\n"
	exit()


entrenar()

