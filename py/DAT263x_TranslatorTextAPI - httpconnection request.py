# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 18:00:25 2018

@author: T901
"""
"""
Go to your Azure portal webpage > Choose All Resources > Choose YOUR Translator Text API service
> Choose Overview to see the Keys > copy it to the transTextKey below
"""
transTextKey = "a3760079619b450caf87854a8e5e5b5b"

import requests, http.client, urllib.request, urllib.parse, urllib.error, base64, json, urllib
from xml.etree import ElementTree

# See language list: https://docs.microsoft.com/en-us/azure/cognitive-services/translator/languages
textToTranslate = input('Please enter some text: \n')
fromLangCode = input('What language is this?: \n') 
toLangCode = input('To what language would you like it translated?: \n') 

try:
    # Connect to server to get the Access Token
    apiKey = transTextKey
    params = ""
    headers = {"Ocp-Apim-Subscription-Key": apiKey}
    AccessTokenHost = "api.cognitive.microsoft.com"
    path = "/sts/v1.0/issueToken"

    conn = http.client.HTTPSConnection(AccessTokenHost)
    conn.request("POST", path, params, headers)
    response = conn.getresponse()
    data = response.read()   # read into byte array
    conn.close()
    accesstoken = "Bearer " + data.decode("UTF-8")  # only valid in 10 minutes  # decode data to string
    print("Access Token = ", accesstoken)

    print("\n===== Method 1 =====")
    # Define the request headers.
    headers = {
        'Authorization': accesstoken
    }

    # Define the parameters
    params = urllib.parse.urlencode({
        "text": textToTranslate,
        "to": toLangCode,
        "from": fromLangCode
    })
    
    print(params)

    # Execute the REST API call and get the response.
    conn = http.client.HTTPSConnection("api.microsofttranslator.com")
    """ Only passing the parameters and empty body with GET METHOD """
    conn.request("GET", "/V2/Http.svc/Translate?%s" % params, "{body}", headers)
    response = conn.getresponse()
    data = response.read()
    translation = ElementTree.fromstring(data.decode("utf-8"))  # Parse XML document from string constant.
    print(translation.text)
    
    print("\n===== Method 2 =====")
    print("Ref: https://docs.microsoft.com/en-us/azure/cognitive-services/translator/quickstarts/python")
    
    host = 'api.microsofttranslator.com'
    path = '/V2/Http.svc/Translate'
    
    params = '?to=' + toLangCode + '&from=' + fromLangCode + '&text=' + urllib.parse.quote (textToTranslate)
    
    def get_suggestions ():
    
        headers = {'Ocp-Apim-Subscription-Key': transTextKey}
        conn = http.client.HTTPSConnection(host)
        conn.request ("GET", path + params, None, headers)
        response = conn.getresponse ()
        return response.read ()
    
    result = get_suggestions ()
    print (result.decode("utf-8"))

    conn.close()
    
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))