# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 10:16:22 2018

@author: T901
"""

import urllib.request
import json

data = {
        "Inputs": {
                "input1":
                [
                    {
                            'PatientID': "1485251",   
                            'Pregnancies': "1",   
                            'PlasmaGlucose': "156",   
                            'DiastolicBloodPressure': "53",   
                            'TricepsThickness': "15",   
                            'SerumInsulin': "226",   
                            'BMI': "29.78619164",   
                            'DiabetesPedigree': "0.203823525",   
                            'Age': "10",   # age 20 gives 0 (non-diabetic), age 45 gives 1
                    }
                ],
        },
    "GlobalParameters":  {
    }
}

body = str.encode(json.dumps(data))  # json.dumps (vs json.loads) : Serialize obj to a JSON formatted str

url = 'https://asiasoutheast.services.azureml.net/subscriptions/ba5540c359f440149fe92a02507cda53/services/d230a2bc980f46589742061e862738e6/execute?api-version=2.0&format=swagger'
api_key = '25j//l4aM97zzM6dZ8zIqRHhRjtMH2f43CIWy4ksgjObcIxB7CfTItxRxOF3PKkkZVX5ekAEtcgjFDUR1Kz6WA==' # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(json.loads(error.read().decode("utf8", 'ignore')))