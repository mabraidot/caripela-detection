# -*- coding: utf-8 -*-
import sys, os
from time import sleep
import cv2
import numpy as np
from FPS.WebcamVideoStream import WebcamVideoStream


# En Linux parece haber un problema con libv4l y se necesita recargarla
#from os import environ
#env = dict(environ)
#env['LD_PRELOAD'] = '/usr/lib/arm-linux-gnueabihf/libv4l/v4l2convert.so'

"""
Función que genera el menú de opciones que guía al usuario para realizar el entrenamiento.
Genera una lista con los nombres conocidos por el modelo y la opción de crear una entrada 
con una cara nueva.
"""
opciones = {}
def menu():

    global opciones

    (nombres, id) = ({}, 0)
    etiqueta = -1
    # Abrimos la lista de nombres conocidos y generamos el menú con el nombre y ID de la persona
    with open('conocidos.csv','r') as csv_nombres:
        for renglon in csv_nombres:
            etiqueta,nombre = renglon.split(',')
            nombre = str.replace(nombre, "\"", "")
            nombre = str.strip(nombre)
            opciones[int(etiqueta)] = nombre
    
    # Agregamos una opción para cargar el nombre de una cara desconocida
    opciones['n'] = "Nuevo rostro"
    
    # Preparamos el texo del menú
    texto_opciones = "\n\nSi deseás actualizar el modelo de una persona que ya existía, seleccioná la opción con su nombre.\nSi es una persona desconocida, presioná n para entrenar el modelo: \n\n"
    for id, texto in opciones.items():
        texto_opciones = texto_opciones + "(" + str(id) + ") " + texto + "\n"
    
    # Mostramos el menú en pantalla y esperamos que el usuario elija una opción
    opcion = input(texto_opciones)
    # Si la opción elegida es salir, salimos
    if opcion == 'q':
        exit()
    # Si eligió nueva cara, usamos el próximo ID disponible y le pedimos al usuario
    # que ingrese el nombre de la nueva cara
    elif opcion == 'n':
        nombre = input('\nIngresá el nombre de la persona y presioná <Enter>: ')
        id = int(etiqueta) + 1
        return id, nombre
    # Si desea actualizar una cara ya conocida con nuevas fotos, tomamos el nombre
    # y ID conocidos
    elif int(opcion) in opciones:
        nombre = opciones[int(opcion)]
        id = int(opcion)
        return id, nombre
    # Si no eligió alguna de las anteriores, mostramos mensaje de error y cerramos el programa
    else:
        print(u"Opción no válida.")
        exit()



"""
Función para realizar el entrenamiento de reconocimiento.
Si hay fotos para reconocer, las usa para entrenar al algoritmo de reconocimiento y actualiza el 
archivo xml de caras conocidas
"""
def entrenar():
    
    global opciones
    
    # Inicializamos algunas variables
    (imagenes, etiquetas, nombres, id) = ([], [], {}, 0)
    BASE_PATH = "Caras/"
    SEPARATOR=";"
    
    """
    Tomamos las fotos dentro de la carpeta definida, guardamos sus datos en un arreglo, 
    y luego las borramos del directorio para liberar espacio.
    """
    for carpeta, subcarpetas, fotos in os.walk(BASE_PATH):
        for foto in fotos:	
            ruta = os.path.join(carpeta, foto)
            if os.path.exists(ruta):
                imagenes.append(cv2.imread(ruta, 0))
                os.remove(ruta)
    
    
    # Si hay imágenes en el directorio, iniciamos el entrenamiento
    if len(imagenes) > 0:
    
        # Mostramos el menú y esperamos la opción elegida por el usuario
        (id, nombre) = menu()
        # Por cada foto encontrada en el directorio, generamos una entrada en el arreglo de las etiquetas
        for etiqueta in imagenes:
            etiquetas.append(int(id))
        
        # Convertimos las imágenes a un array numpy, que es lo que el algoritmo de reconocimiento necesita
        (imagenes, etiquetas) = [np.array(lista) for lista in [imagenes, etiquetas]]
        
        
        # Inicializamos el modelo
        modelo = cv2.face.LBPHFaceRecognizer_create()
        archivo_modelo = 'conocidos.xml'
        # Si ya existía el archivo xml de reconocimiento, lo actualizamos
        if os.path.exists(archivo_modelo):
            modelo.read(archivo_modelo)
            modelo.update(imagenes, etiquetas)
            print(u'Actualizando el modelo...')
        # Si no existe el archivo xml de reconocimiento, lo creamos
        else:
            modelo.train(imagenes, etiquetas)
            print(u'Creando un nuevo modelo...')
        
        # Guardamos los cambios en el modelo
        modelo.write(archivo_modelo)
        # Si entrenamos una nueva cara, agregamos su nombre y ID a la lista de personas conocidas
        if int(id) not in opciones:
            csv_nombres = open('conocidos.csv', 'a+')
            csv_nombres.write(str(id)+', "'+nombre+'"\n')
            csv_nombres.close()
            print(u'Entrenando la nueva cara de ' + nombre)
        else:
            print(u'Actualizando la cara de ' + nombre)
        
        
    # Si no hay imágenes en el directorio, mostramos el error y cerramos el programa
    else:
        os.system('espeak -ves+f1 -s130 "c No hay imágenes disponibles para realizar el entrenamiento" 2>/dev/null')
        print(u"\n\nNo hay imágenes disponibles para realizar el entrenamiento.\n\n")
    exit()

# Llamada a la función de entrenamiento
entrenar()

