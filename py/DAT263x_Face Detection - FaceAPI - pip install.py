# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 10:08:00 2018

@author: T901
"""
"""
From Azure portal, create Face API service
After created, go to the service, look for Overview and Keys for Endpoint URL and Key
"""

faceURI = "https://southeastasia.api.cognitive.microsoft.com/face/v1.0"
faceKey = "3d8401a023164bbda33a1f4b7e71c3a2"

import pip

def install(package):
    pip.main(['install', package])

try:
    import cognitive_face
#except ImportError as e:
except Exception as e:
    print("Fail to install cognitive_face: ", e)
    install('cognitive_face')
    
try: 
    import pillow
except Exception as e:
    print("Fail to install pillow: ", e)
    install('pillow')    

print("\n-Detect a face in an image")
#%matplotlib inline
import requests
from io import BytesIO
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cognitive_face as CF

# Set URI and Key
CF.Key.set(faceKey)
CF.BaseUrl.set(faceURI)

# Detect faces in an image
img_url = 'https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/graeme1.jpg'
result = CF.face.detect(img_url)
print("result = ", result)

# Get the ID of the first face detected
face1 = result[0]['faceId']
print ("Face 1 ID: " + face1)

# Get the image
response = requests.get(img_url)                # grab the image
img = Image.open(BytesIO(response.content))     # open it up
print("Type of img after doing Image.open: ", type(img))

# Add rectangles for each face found
color="blue"
if result is not None:
    draw = ImageDraw.Draw(img) 
    for currFace in result:
        faceRectangle = currFace['faceRectangle']
        left = faceRectangle['left']
        top = faceRectangle['top']
        width = faceRectangle['width']
        height = faceRectangle['height']
        draw.line([(left,top),(left+width,top)],fill=color, width=5)
        draw.line([(left+width,top),(left+width,top+height)],fill=color , width=5)
        draw.line([(left+width,top+height),(left, top+height)],fill=color , width=5)
        draw.line([(left,top+height),(left, top)],fill=color , width=5)

# show the image
imshow(img)  
plt.show()  

print("\n-Compare with another image")
# Get the image to compare
img2_url = 'https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/graeme2.jpg'
response2 = requests.get(img2_url)
img2 = Image.open(BytesIO(response2.content))

# Detect faces in a comparison image
result2 = CF.face.detect(img2_url)

# Assume the first face is the one we want to compare
face2 = result2[0]['faceId']
print ("Face 2 ID :" + face2)

def verify_face(face1, face2):
    # By default, assume the match is unverified
    verified = "Not Verified"
    color="red"

    if result2 is not None:
        # compare the comparison face to the original one we retrieved previously
        verify = CF.face.verify(face1, face2)
        print("verify = ", verify)

        # if there's a match, set verified and change color to green
        if verify['isIdentical'] == True:
            verified = "Verified"
            color="lightgreen"

        # Display the second face with a red rectange if unverified, or green if verified
        draw = ImageDraw.Draw(img2) 
        for currFace in result2:
            faceRectangle = currFace['faceRectangle']
            left = faceRectangle['left']
            top = faceRectangle['top']
            width = faceRectangle['width']
            height = faceRectangle['height']
            draw.line([(left,top),(left+width,top)] , fill=color, width=5)
            draw.line([(left+width,top),(left+width,top+height)] , fill=color, width=5)
            draw.line([(left+width,top+height),(left, top+height)] , fill=color, width=5)
            draw.line([(left,top+height),(left, top)] , fill=color, width=5)

    # show the image
    imshow(img2)
    plt.show()

    # Display verification status and confidence level
    print(verified)
    print ("Confidence Level: " + str(verify['confidence']))

verify_face(face1, face2)

print("\n-Check with another person")
# Get the image to compare
img2_url = 'https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/graeme3.jpg'
response2 = requests.get(img2_url)
img2 = Image.open(BytesIO(response2.content))

# Detect faces in a comparison image
result2 = CF.face.detect(img2_url)

# Assume the first face is the one we want to compare
face2 = result2[0]['faceId']
print ("Face 3 ID: " + face2)

verify_face(face1, face2)

print("\n-Compare with one more image")
# Get the image to compare
img2_url = 'https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/satya.jpg'
response2 = requests.get(img2_url)
img2 = Image.open(BytesIO(response2.content))

# Detect faces in a comparison image
result2 = CF.face.detect(img2_url)

# Assume the first face is the one we want to compare
face2 = result2[0]['faceId']
print ("Face 4:" + face2)

verify_face(face1, face2)