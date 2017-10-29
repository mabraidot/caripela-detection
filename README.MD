CARIPELA DETECTION
====================
IFTS N° 14 - Tecnicatura Superior en Robótica
---------------------------------------------

**UBICACION DEL PROYECTO**
* ~/Documents/Proyectos/caripela_detection/

**ESTRUCTURA DE ARCHIVOS Y DIRECTORIOS**
* **/Caras/**: Almacena las imágenes tomadas por la cámara, que se utilizarán para entrenar al algoritmo de reconocimiento.
* **/FPS/**: Librería para utilizar el resto de los núcleos del procesador.
* **/haar/**: Archivos preentrenador para reconocer una cara dentro de una imágen.
* **/caripela_recognition.py**: Intenta reconocer e identificar una cara dentro del video. Si no es conocida, le toma fotos para usar en el algoritmo de entrenamiento.
* **/caripela_training.py**: Si hay fotos de una persona desconocida, las usa para entrenar al algoritmo de identificación.
* **conocidos.xml**: Almacena las características de las caras pre-entrenadas que utilizará el algoritmo de identificación.
* **conocidos.csv**: Almacena los nombres de las caras conocidas.

**MODO DE USO - INICIALIZACIÓN**
* cd ~/Documents/Proyectos/caripela_detection/
* source ~/.profile
* workon cv3

*Solo debe ejecutarse una vez, al iniciar Raspbian. Esto inicia un ambiente virtual de python, el mismo ya tiene instaladas las librerías necesarias para el funcionamiento del proyecto.*

**MODO DE USO EN RECONOCIMIENTO**
* python caripela_recognition.py 

*Este programa inicia la ventana de video solicitando que el usuario se ubique frente a la cámara para iniciar el reconocimiento. Solo tomará caras que tengan más de 250 px de ancho, por lo que el sujeto debe ubicarse relativamente cerca de la cámara.

Si reconoce una cara, dibuja un recuadro donde está ubicada, e intenta identificarla usando el archivo de caras pre-entrenadas. Si ya estaba entrenada, es decir es una cara conocida, muestra el nombre de la persona en el recuadro, junto con el índice de confianza en la predicción (un número menor indica más confianza).
Si no es una cara conocida, toma 20 fotos separadas por un número configurable de fotogramas, para luego utilizarlas en el programa de entrenamiento.*

**MODO DE USO EN ENTRENAMIENTO**
* python caripela_training.py 

*Toma las fotos ubicadas en el directorio __Caras__, y solicita al usuario que indique si se va a realizar una actualización del entrenamiento de una cara conocida, o se se va a agregar un a nueva cara. En este caso se solicitará el nombre de la persona a reconocer.*