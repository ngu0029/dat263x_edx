# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 15:24:31 2018

@author: T901
"""
"""
Video Indexer AI
https://www.videoindexer.ai/ 
(Choosing Sign in with a Live or Outlook account with gmail user dungnq.vtrd@gmail.com
or You can sign up and log in with google account dungnq.vtrd@gmail.com to sync with videobreakdown portal account below)

===========================================
Face Tracking
Create a Video Indexer API Subscription
https://videobreakdown.portal.azure-api.net/ (You can sign up and log in with google account dungnq.vtrd@gmail.com)
- Log in
- Choose Products > Free Preview
- Make a subscription
- Go to the subscription above to see Key as the apiKey below
"""

apiKey = "b187b543ef1e4cdaa469d88c12215d2e"

print("\n-Upload a Video for Processing")
import http.client, urllib.request, urllib.parse, urllib.error, base64

# We'll upload this video from GitHub to the Video Indexer
video_url='https://github.com/MicrosoftLearning/AI-Introduction/raw/master/files/vid.mp4'

headers = {
    # Request headers
    'Content-Type': 'multipart/form-data',
    'Ocp-Apim-Subscription-Key': apiKey,
}

params = urllib.parse.urlencode({
    # Request parameters
    'name': 'vid',                  # name to assign to the video
    'privacy': 'Private',           # use Private for privacy level
    'videoUrl': video_url,
    'language': 'en-US',
})

try:
    conn = http.client.HTTPSConnection('videobreakdown.azure-api.net')
    conn.request("POST", "/Breakdowns/Api/Partner/Breakdowns?%s" % params, "{body}", headers)
    response = conn.getresponse()
    data = response.read()
    print("response data = ", data)
    # The response is a byte array - convert to a string and remove quotes
    vidId = data.decode("UTF-8").replace('"', '')
    print("Breakdown ID: " + vidId)
    conn.close()
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))
    
print("\n-Check status")
headers = {
    # Request headers
    'Ocp-Apim-Subscription-Key': apiKey,
}

params = urllib.parse.urlencode({
})

try:
    conn = http.client.HTTPSConnection('videobreakdown.azure-api.net')
    conn.request("GET", "/Breakdowns/Api/Partner/Breakdowns/%s/State?%s" % (vidId, params), "{body}", headers)
    response = conn.getresponse()
    data = response.read()
    print(data)
    conn.close()
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))
    
print("\n-View the video")
from IPython.core.display import HTML

headers = {
    # Request headers
    'Ocp-Apim-Subscription-Key': apiKey,
}

params = urllib.parse.urlencode({
})

try:
    conn = http.client.HTTPSConnection('videobreakdown.azure-api.net')
    conn.request("GET", "/Breakdowns/Api/Partner/Breakdowns/%s/PlayerWidgetUrl?%s" % (vidId, params), "{body}", headers)
    response = conn.getresponse()
    data = response.read()          # byte array
    vidUrl = data.decode("UTF-8").replace('"', '')  # convert to a string and remove quotes
    print (vidUrl)
    conn.close()
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))
    
HTML('<iframe width=600 height=400 src="%s"/>' % vidUrl )

print("\n-View the video breakdown")
import json

headers = {
    # Request headers
    'Ocp-Apim-Subscription-Key': apiKey,
}

params = urllib.parse.urlencode({
    # Request parameters
    'language': 'en-US',
})

try:
    conn = http.client.HTTPSConnection('videobreakdown.azure-api.net')
    conn.request("GET", "/Breakdowns/Api/Partner/Breakdowns/%s?%s" % (vidId, params), "{body}", headers)
    response = conn.getresponse()
    data = response.read()          # byte array
    strData = data.decode("UTF-8")  # convert to string
    jData = json.loads(strData)     # convert json document to python object
    print (json.dumps(jData, sort_keys=True, indent=2)) # dump back to json document
    conn.close()
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))

print("\n-Get Details of Faces identified in the Video")
print(json.dumps(jData["summarizedInsights"]["faces"], sort_keys=True, indent=2))
faceUrl = jData["summarizedInsights"]["faces"][0]["thumbnailFullUrl"]
print(faceUrl)

print("-Method 1: Show image by HTML tag img")
HTML('<img src="%s"/>' % faceUrl )

print("-Method 2: Show image by known libraries")
import requests
from PIL import Image
from io import BytesIO
from mathplotlib.pyplot import imshow

response = requests.get(faceUrl)
img = Image.open(BytesIO(response.content))

imshow(img)

print("\n-View and Edit People Insights")
headers = {
    # Request headers
    'Ocp-Apim-Subscription-Key': apiKey,
}

params = urllib.parse.urlencode({
    # Request parameters
    'widgetType': 'People',
    'allowEdit': 'True',
})

try:
    conn = http.client.HTTPSConnection('videobreakdown.azure-api.net')
    conn.request("GET", "/Breakdowns/Api/Partner/Breakdowns/%s/InsightsWidgetUrl?%s" % (vidId, params), "{body}", headers)
    response = conn.getresponse()
    data = response.read()
    iUrl = data.decode("UTF-8").replace('"', '')
    print(iUrl)
    conn.close()
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))

HTML('<iframe width=800 height=600 src="%s"/>' % iUrl )

print("\nReload BreakDown and Check Updated Face Details")
import http.client, urllib.request, urllib.parse, urllib.error, base64, json

headers = {
    # Request headers
    'Ocp-Apim-Subscription-Key': apiKey,
}

params = urllib.parse.urlencode({
    # Request parameters
    'language': 'en-US',
})

try:
    conn = http.client.HTTPSConnection('videobreakdown.azure-api.net')
    conn.request("GET", "/Breakdowns/Api/Partner/Breakdowns/%s?%s" % (vidId, params), "{body}", headers)
    response = conn.getresponse()
    data = response.read()
    strData = data.decode("UTF-8")
    jData = json.loads(strData)
    print(json.dumps(jData["summarizedInsights"]["faces"], sort_keys=True, indent=2))
    conn.close()
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))