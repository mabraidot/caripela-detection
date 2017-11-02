# import the necessary packages
from threading import Thread
import sys, os

class ESpeak:
    def __init__(self):
        

    def decir(self, texto):
        # Comenzar el hilo e indicarle la función que ejecutará
        Thread(target=self.update, args=(texto)).start()
        return self

    def update(self, texto):
	os.system('espeak -ves+f1 -s130 "'+texto+'" 2>/dev/null')

