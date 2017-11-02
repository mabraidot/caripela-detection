# import the necessary packages
from threading import Thread
import sys, os

class ESpeak:
    def decir(self, texto):
        # Comenzar el hilo e indicarle la función que ejecutará
        Thread(target=self.update, args=[texto]).start()
        return self

    def update(self, texto):
        os.system('espeak -ves+f4 -s120 "'+texto+'" 2>/dev/null')

