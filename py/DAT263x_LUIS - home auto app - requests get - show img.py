# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 12:10:33 2018

@author: T901
"""

#%matplotlib inline
from matplotlib.pyplot import imshow
from PIL import Image
import requests
from io import BytesIO
import json 

"""
- Log in https://www.luis.ai/home
- Design and build the Home Automation app (create intents, utterances, entities)
- Train and test it
- In Publish page, Add Key to assign subscription key to your app before publishing it to production slot
- Copy the endpoint url created by Add Key in the previous step to the endpointUrl variable below
"""

# Set up API configuration
endpointUrl = "https://southeastasia.api.cognitive.microsoft.com/luis/v2.0/apps/eddf63ca-aa2c-455d-8a79-8999cee4633d?subscription-key=c1c13d5ba4c5477ea03c1b1a60b6ba3b&verbose=true&timezoneOffset=420&q="

# prompt for a command
command = input('Please enter a command: \n')

# Call the LUIS service and get the JSON response
endpoint = endpointUrl + command.replace(" ","+")
response = requests.get(endpoint)
print(response)
#data = json.loads(response.content.decode("UTF-8"))  # OK # # json.loads (vs json.dumps) : Deserialize s (a str, bytes or bytearray instance containing a JSON document) to a Python object
data = response.json() # OK, see file DAT263x_TextAnalyticsAPI - QuickStart.py
print(data)

# Identify the top scoring intent
intent = data["topScoringIntent"]["intent"]
if (intent == "Light On"):
    img_url = 'https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/LightOn.jpg'
elif (intent == "Light Off"):
    img_url = 'https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/LightOff.jpg'
else:
    img_url = 'https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/Dunno.jpg'

# Get the appropriate image and show it
response = requests.get(img_url)
img = Image.open(BytesIO(response.content))
imshow(img)
