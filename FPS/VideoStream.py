# import the necessary packages
from .WebcamVideoStream import WebcamVideoStream

class VideoStream:
    def __init__(self, src=-1, usePiCamera=False, resolution=(320, 240), framerate=30):
        # check to see if the picamera module should be used
        if usePiCamera:
            # only import the picamera packages unless we are
            # explicity told to do so -- this helps remove the
            # requirement of `picamera[array]` from desktops or
            # laptops that still want to use the `imutils` package
            from .PiVideoStream import PiVideoStream
            # initialize the picamera stream and allow the camera
            # sensor to warmup
            self.stream = PiVideoStream(resolution=resolution, framerate=framerate)

        # otherwise, we are using OpenCV so initialize the webcam stream
        else:
            self.stream = WebcamVideoStream(src=src, resolution=resolution)

    def start(self):
        # start the threaded video stream
        return self.stream.start()
 
    def update(self):
        # grab the next frame from the stream
        self.stream.update()
 
    def read(self):
        # return the current frame
        return self.stream.read()
 
    def stop(self):
        # stop the thread and release any resources
        self.stream.stop()