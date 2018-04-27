# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 16:58:28 2018

@author: T901
"""
"""
Computer Vision API
https://www.customvision.ai/projects
- Create new project, choose domain
- Add images and corresponding tags
- Train the images (it will withhold some images for testing)
- Train OK then Make as default
- Click on Prediction URL to find Project ID and Prediction Key for your request below
"""

#%matplotlib inline
from matplotlib.pyplot import imshow
from PIL import Image
import requests
from io import BytesIO
import http.client, urllib.request, urllib.parse, urllib.error, base64, json

img_url = 'https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/Vision/Test.jpg'

headers = {
    # Request headers
    'Content-Type': 'application/json',
    'Prediction-key': 'ffed3de839254548bfc7b0fc69e1db5b',
}

params = urllib.parse.urlencode({
})

body = "{'Url':'" + img_url + "'}"

try:
    # Get the predicted tags
    conn = http.client.HTTPSConnection('southcentralus.api.cognitive.microsoft.com')
    conn.request("POST", "/customvision/v1.0/Prediction/b1848ff8-33b5-4bae-86fd-959365906c94/url?%s" % params, body, headers)
    response = conn.getresponse()
    data = response.read()              
    strData = data.decode("UTF-8")      # convert byte array to string
    parsed = json.loads(strData)
    print(parsed)
    
    # sort the tags by probability and get the highest one
    sorted_predictions = dict(parsed)    # parsed already a dict
    print(sorted_predictions)
    sorted_predictions['Predictions'] = sorted(parsed['Predictions'], key=lambda x : x['Probability'], reverse=True)
    print(sorted_predictions['Predictions'][0]['Tag'])
    conn.close()

    # Get the image and show it
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    imshow(img)
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))
    