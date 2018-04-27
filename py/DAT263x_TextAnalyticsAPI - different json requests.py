# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 12:51:52 2018

@author: T901
"""

"""
Go to your Azure portal webpage > Choose All Resources > Choose YOUR Text Analytics service
> Choose Overview to see the Endpoint > copy it to the textAnalyticsURI below
"""
textAnalyticsURI = 'https://southeastasia.api.cognitive.microsoft.com/text/analytics/v2.0'
textKey = '482fc455fc0744c399186303e64218f4'

import http.client, urllib.request, urllib.parse, urllib.error, base64, json, urllib

"""
See: https://stackoverflow.com/questions/22676/how-do-i-download-a-file-over-http-using-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
"""
#!curl https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/Gettysburg.txt -o Gettysburg.txt
urllib.request.urlretrieve('https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/Gettysburg.txt', 'Gettysburg.txt')
doc2 = open("Gettysburg.txt", "r")
doc2Txt = doc2.read()
#print (doc2Txt)

#!curl https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/Cognitive.txt -o Cognitive.txt
doc3 = urllib.request.urlretrieve('https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/Cognitive.txt','Cognitive.txt')
doc3 = open("Cognitive.txt", "r")
doc3Txt = doc3.read()
#print (doc3Txt)

# Define the request headers.
headers = {
    'Content-Type': 'application/json',
    'Ocp-Apim-Subscription-Key': textKey,
    'Accept': 'application/json'
}

# Define the parameters
params = urllib.parse.urlencode({
})

# Define the request body
body = {
  "documents": [
    {
        "language": "en",
        "id": "1",
        "text": doc2Txt
    },
    {
        "language": "en",
        "id": "2",
        "text": doc3Txt
    }
  ]
}

try:
    print("""===== METHOD 1 =====""")
    """
    From https://azure.microsoft.com/en-us/services/cognitive-services/text-analytics/
    > Click on API or API Reference
    > Click on Key Phrases
    > Scroll down to see Code Samples for Python
    """
    # Execute the REST API call and get the response.
    conn = http.client.HTTPSConnection('southeastasia.api.cognitive.microsoft.com')
    """ Similar for Detect Language, Key Phrases, Sentiment """
    conn.request("POST", "/text/analytics/v2.0/keyPhrases?%s" % params, str(body), headers)
    response = conn.getresponse()
    data = response.read().decode("UTF-8") # OK! # read() into byte array, then convert to string
    #data = response.read() # OK!
    print(data)

    # 'data' contains the JSON response, which includes a collection of documents.
    parsed = json.loads(data)  # json.loads (vs json.dumps) : Deserialize s (a str, bytes or bytearray instance containing a JSON document) to a Python object
    print("\nparsed = ", parsed)
    print("---------------------------")
    for document in parsed['documents']:
        print("Document " + document["id"] + " key phrases:")
        for phrase in document['keyPhrases']:
            print("  " + phrase)
        print("---------------------------")
    conn.close()
    
    print("""\n===== METHOD 2 =====""")
    # See: Quickstart for Text Analytics API with Python
    print("Ref: https://docs.microsoft.com/en-us/azure/cognitive-services/Text-Analytics/QuickStarts/Python")
    import requests
    from pprint import pprint
    
    key_phrase_api_url = textAnalyticsURI + "/keyPhrases"
    print(key_phrase_api_url)

    headers   = {"Ocp-Apim-Subscription-Key": textKey}
    response  = requests.post(key_phrase_api_url, headers=headers, json=body)
    key_phrases = response.json()
    pprint(key_phrases)
    
except Exception as e:
    print('Error:')
    print(e)