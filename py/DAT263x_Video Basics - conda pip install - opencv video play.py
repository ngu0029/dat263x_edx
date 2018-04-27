# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 11:31:45 2018

@author: T901
"""

"""
This video is really just a stream of static images known as frames 
and they're encoded in a format specific codec and 
encapsulated in a container which also includes a header with
metadata about the format, duration and so on
"""

print("\n-Install av package with conda")
import conda.cli

"""
Ref: https://stackoverflow.com/questions/41767340/using-conda-install-within-a-python-script?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
"""
def install_conda(package, option):
    #conda.cli.main('conda', 'install', package, '-c', 'conda-forge', '-y')
    conda.cli.main('conda', 'install', package, option)
 
    
import pip    
    
def install_pip(package, option):
    if option is None:
        pip.main(['install', package])
    else:    
        pip.main(['install', package, option])
    
try:
    import av
except Exception as e:
    print("Fail to install av: ", e)
    install_conda('av', '-c conda-forge -y')
    import av

print("\n-Grab video")
import urllib.request

urllib.request.urlretrieve('https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/Intro.mp4', 'video.mp4')

print("\n-Install opencv package with pip and play video")
"""
IPYTHON NOTEBOOK CODE
%%HTML
<video width="320" height="240" controls>
  <source src="video.mp4" type="video/mp4">
</video>
"""
import numpy as np

"""
Ref: https://www.scivision.co/install-opencv-python-windows/
https://stackoverflow.com/questions/23119413/how-do-i-install-python-opencv-through-conda?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
"""
try:
    import cv2
except Exception as e:
    print("Fail to install opencv: ", e)
    #install_conda('opencv', '-c conda-forge')
    #install_conda('opencv', '-c menpo')
    install_pip('opencv-python', None)    
    import cv2  

"""
Ref: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html 
     https://stackoverflow.com/questions/33900546/playing-video-file-using-python-opencv?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
"""
cap = cv2.VideoCapture('video.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret == True:
        cv2.imshow('frame', frame)
        # & 0xFF is required for a 64-bit system
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

print("\n-Count frames and show the 25th one")
#import av
container = av.open('video.mp4')  # all image frames encoded and encapsulated in a container

#%matplotlib inline
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw

for frame in container.decode(video=0):  # the first of video codecs
    if (frame.index == 25):
        img = frame.to_image()
        imshow(img)
frameCount = frame.index - 1
print(str(frameCount) + " frames")