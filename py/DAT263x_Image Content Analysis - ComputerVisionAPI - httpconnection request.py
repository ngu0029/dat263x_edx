# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 09:18:17 2018

@author: T901
"""
"""
From your Azure portal, creat Computer Vision API service
After created, go to the service, look for Overview and Keys for Endpoint URL and Key
"""

visionURI = 'southeastasia.api.cognitive.microsoft.com'
visionKey = '672128ec71ff4f258db9d154ffb8c87e'

#%matplotlib inline
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

img_url = 'https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/graeme2.jpg'

# Get the image and show it
response = requests.get(img_url)
img = Image.open(BytesIO(response.content))
imshow(img)
plt.show()

print("\n-Use the Computer Vison API to get image features")
def get_image_features(img_url):
    import http.client, urllib.request, urllib.parse, urllib.error, base64, json

    headers = {
        # Request headers.
        'Content-Type': 'application/json',
        'Ocp-Apim-Subscription-Key': visionKey,
    }

    params = urllib.parse.urlencode({
        # Request parameters. All of them are optional.
        'visualFeatures': 'Categories,Description,Color',  # features to get back
        'language': 'en',                                  # language to get back
    })

    body = "{'url':'" + img_url + "'}"

    try:
        # Execute the REST API call and get the response.
        conn = http.client.HTTPSConnection(visionURI)
        conn.request("POST", "/vision/v1.0/analyze?%s" % params, body, headers)
        response = conn.getresponse()
        data = response.read()

        # 'data' contains the JSON response.
        parsed = json.loads(data)  # Deserialize a JSON document to a Python object
        if response is not None:
            return parsed
        conn.close()


    except Exception as e:
        print('Error:')
        print(e)
        
jsonData = get_image_features(img_url)
desc = jsonData['description']['captions'][0]['text']
print(desc)

print("\n-Get the full response")
# View the full details returned
import http.client, urllib.request, urllib.parse, urllib.error, base64, json
print (json.dumps(jsonData, sort_keys=True, indent=2))  # Serialize obj to a JSON formatted str.

print("\n-Let's try with another image")
#img_url = 'https://raw.githubusercontent.com//MicrosoftLearning/AI-Introduction/master/files/uke.jpg'
img_url = 'https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/Vision/Test.jpg'
# Get the image and show it
response = requests.get(img_url)                # grab the image
img = Image.open(BytesIO(response.content))     # open it up
imshow(img)
plt.show()
jsonData = get_image_features(img_url)
desc = jsonData['description']['captions'][0]['text']
print(desc)

print("\n-How about something a little more complex")
img_url = 'https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/soccer.jpg'

# Get the image and show it
response = requests.get(img_url)
img = Image.open(BytesIO(response.content))
imshow(img)
jsonData = get_image_features(img_url)
desc = jsonData['description']['captions'][0]['text']
print(desc)
