# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 14:09:22 2018

@author: T901
"""
"""
Go to your Azure portal webpage > Choose All Resources > Choose YOUR Bing Speech API service
> Choose Keys to see the subscription key > copy it to the speechKey below
"""

speechKey = '443d193a670144c7b13609bdb50db3bc'

import IPython
import http.client, urllib.parse, json
from xml.etree import ElementTree

# Get the input text
myText = input('What would you like me to say?: \n')

# The Speech API requires an access token (valid for 10 mins)
apiKey = speechKey
params = ""
headers = {"Ocp-Apim-Subscription-Key": apiKey}
# Bing Speech API service enpoint: https://api.cognitive.microsoft.com/sts/v1.0
AccessTokenHost = "api.cognitive.microsoft.com"
path = "/sts/v1.0/issueToken"

# Use the API key to request an access token
conn = http.client.HTTPSConnection(AccessTokenHost)
conn.request("POST", path, params, headers)
response = conn.getresponse()
data = response.read()
conn.close()
accesstoken = data.decode("UTF-8")      # only valid in 10 minutes
print("Access Token = ", accesstoken)

# Now that we have a token, we can set up the request
""" RATHER THAN JSON, USING XML ELEMENT TO SET THINGS UP """
"""
XML Element is a flexible container object designed to store hierarchical data structures in memory. 
It can be described as a cross between a list and a dictionary. 

ElementTree: Lightweight XML support for Python.
You can also use the ElementTree class to wrap an element structure 
and convert it to and from XML.
"""
body = ElementTree.Element('speak', version='1.0')
# Equivalent to element_attrib[key] = value
body.set('{http://www.w3.org/XML/1998/namespace}lang', 'en-us')
voice = ElementTree.SubElement(body, 'voice')  # voice is a sub-element of the body element
voice.set('{http://www.w3.org/XML/1998/namespace}lang', 'en-US')
voice.set('{http://www.w3.org/XML/1998/namespace}gender', 'Male')
voice.set('name', 'Microsoft Server Speech Text to Speech Voice (en-US, JessaRUS)')
voice.text = myText
headers = {"Content-type": "application/ssml+xml", 
           "X-Microsoft-OutputFormat": "riff-16khz-16bit-mono-pcm",   # 16 bit mono audio stream
           "Authorization": "Bearer " + accesstoken, 
           "X-Search-AppId": "07D3234E49CE426DAA29772419F436CA", 
           "X-Search-ClientID": "1ECFAE91408841A480F00935DC390960", 
           "User-Agent": "TTSForPython"}

#Connect to server to synthesize a wav from the text
conn = http.client.HTTPSConnection("speech.platform.bing.com")
conn.request("POST", "/synthesize", ElementTree.tostring(body), headers)
response = conn.getresponse()
data = response.read()
print(type(data), len(data))
conn.close()

#Play the wav
#IPython.display.Audio(data, autoplay=True)
#import pip
#
#pip.main(['install', 'playsound'])
#
#from playsound import playsound
#
#playsound(data)

#import pip
#
#pip.main(['install', 'sounddevice'])
#
#import sounddevice as sd
#
#sd.play(data)

"""
X-Microsoft-OutputFormat: riff-16khz-16bit-mono-pcm
https://github.com/Azure-Samples/Cognitive-Speech-TTS/wiki/how-to-choose-different-audio-output-format
"""
#import pip
#
#pip.main(['install', 'ao'])
#
#import ao
#from ao import AudioDevice
#
#dev = AudioDevice(2, bits=16, rate=16000,channels=1)
#dev.play(data)

"""
X-Microsoft-OutputFormat: riff-16khz-16bit-mono-pcm
https://stackoverflow.com/questions/8707967/playing-a-sound-from-a-wave-form-stored-in-an-array?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
"""
import pyaudio

# instantiate PyAudio (1)
p = pyaudio.PyAudio()

# open stream (2), 2 is size in bytes of int16
stream = p.open(format=p.get_format_from_width(2),
                channels=1,   # mono
                rate=16000,
                output=True)

# play stream (3), blocking call
stream.write(data)

# stop stream (4)
stream.stop_stream()
stream.close()

# close PyAudio (5)
p.terminate()