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
	
	(images, labels, names, id) = ([], [], {}, 0)
	with open('conocidos.csv','r') as f:
		for line in f:
			label,name = line.split(',')
			#names[int(label)] = str(name).rstrip('\n').strip().strip('"')
			id = int(label)+1

	
	nombre = raw_input('\nIngrese el nombre de la persona y presione <Enter>: ')	
	BASE_PATH = "Caras"
	SEPARATOR=";"
	names[id] = nombre
	label = id
	for dirname, dirnames, filenames in os.walk(BASE_PATH):
		#for filename in os.listdir(dirname):
		for filename in filenames:
			abs_path = "%s/%s" % (dirname, filename)
			#print "%s%s%d" % (abs_path, SEPARATOR, label)
			images.append(cv2.imread(abs_path, 0))
			labels.append(int(label))

	
	(images, labels) = [np.array(lis) for lis in [images, labels]]
	model = cv2.createLBPHFaceRecognizer()
	modelFile = 'conocidos.xml'
	if os.path.exists(modelFile):
		model.load(modelFile)
		model.update(images, labels)
		print 'actualizando...'
	else:
		model.train(images, labels)
		print 'actualizando...'
	model.save(modelFile)
	f = open('conocidos.csv', 'a+')
	f.write(str(id)+', "'+nombre+'"\n')
	f.close()
	exit()


def inicio():
	opcion = raw_input('\nPresione (t: entrenar): ')
	if opcion == 't':
		entrenar()
	else:
		inicio()
	return


inicio()

