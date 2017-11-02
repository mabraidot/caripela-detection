CARIPELA DETECTION - INSTALACION
================================

**INSTALACION DE OPENCV EN RASPBERRY PI 3**

* Expandir el filesystem
~~~
$ sudo raspi-config
$ sudo reboot
~~~

* Instalar dependencias
~~~
$ sudo apt-get update && sudo apt-get upgrade
$ sudo apt-get install build-essential cmake pkg-config
$ sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
$ sudo apt-get install libxvidcore-dev libx264-dev
$ sudo apt-get install libgtk2.0-dev libgtk-3-dev
$ sudo apt-get install libcanberra-gtk*
$ sudo apt-get install libatlas-base-dev gfortran
$ sudo apt-get install python2.7-dev python3-dev
~~~

* Bajar código fuente de OpenCV
~~~
$ cd ~
$ wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.3.0.zip
$ unzip opencv.zip
$ wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.3.0.zip
$ unzip opencv_contrib.zip
~~~

* Instalar la herramienta de ambiente virtual para python
~~~
$ wget https://bootstrap.pypa.io/get-pip.py
$ sudo python get-pip.py
$ sudo python3 get-pip.py
$ sudo pip install virtualenv virtualenvwrapper
$ sudo rm -rf ~/.cache/pip
~~~

* Editar el archivo **~/.profile**
~~~
$ vim ~/.profile
~~~

* Pegar lo siguiente al final del archivo
~~~
# virtualenv and virtualenvwrapper
export WORKON_HOME=$HOME/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh
~~~

* Recargar el archivo
~~~
$ source ~/.profile
~~~

* Crear un ambiente virtual
~~~
$ mkvirtualenv cv3 -p python3
~~~

* Instalar la librería numpy
~~~
$ pip install numpy
~~~

* Compilar e instalar OpenCV
~~~
$ workon cv3
$ cd ~/opencv-3.3.0/
$ mkdir build
$ cd build
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.3.0/modules \
    -D ENABLE_NEON=ON \
    -D ENABLE_VFPV3=ON \
    -D BUILD_TESTS=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D BUILD_EXAMPLES=OFF ..
~~~

* Aumentar momentáneamente la cantidad de memoria swap
* Editar en el archivo **/etc/dphys-swapfile**, la variable **CONF_SWAPSIZE**
~~~
...
# set size to absolute value, leaving empty (default) then uses computed value
#   you most likely don't want this, unless you have an special disk situation
# CONF_SWAPSIZE=100
CONF_SWAPSIZE=1024
...
~~~

* Reiniciar el servicio swap
~~~
$ sudo /etc/init.d/dphys-swapfile stop
$ sudo /etc/init.d/dphys-swapfile start
~~~

* Compilar
~~~
$ make -j4
$ sudo make install
$ sudo ldconfig
~~~

* Volver la variable **CONF_SWAPSIZE** a 100 Mb y reiniciar el servicio swap

* Finalizar la instalación creando un link a la librería OpenCV. _**Nota:** los paths contienen puthon3.5, adaptar a la versión instalada._
~~~
$ cd /usr/local/lib/python3.5/site-packages/
$ sudo mv cv2.cpython-35m-arm-linux-gnueabihf.so cv2.so
$ cd ~/.virtualenvs/cv3/lib/python3.5/site-packages/
$ ln -s /usr/local/lib/python3.5/site-packages/cv2.so cv2.so
~~~

* Probar la instalación
~~~
$ source ~/.profile
$ workon cv3
$ python
>>> import cv2
>>> cv2.__version__
'3.3.0'
>>>
~~~

**INSTALACION DE LA LIBRERÍA DE LA CÁMARA**

* Habilitar la cámara desde las preferencias del sistema
* Instalar la librería picamera
~~~
$ source ~/.profile
$ workon cv3
$ pip install "picamera[array]"
~~~
