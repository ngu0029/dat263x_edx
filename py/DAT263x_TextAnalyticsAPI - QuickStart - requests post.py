# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 15:38:44 2018

@author: T901

From https://azure.microsoft.com/en-us/services/cognitive-services/, click on "Try Cognitive Services for free"
> Choose Text Analytics > Get API Key

Quickstart for Text Analytics API with Python
https://docs.microsoft.com/en-us/azure/cognitive-services/Text-Analytics/QuickStarts/Python

Easily evaluate sentiment and topics to understand what users want
5,000 transactions per month.

Endpoint: https://westcentralus.api.cognitive.microsoft.com/text/analytics/v2.0

Key 1: 17c196ec5a014ad18eba26267b6904bb
Key 2: 5a685f02422b4608896e729fc13ced98
"""

print(""" 1. Detect languages """)
subscription_key = '17c196ec5a014ad18eba26267b6904bb'
assert subscription_key

text_analytics_base_url = "https://westcentralus.api.cognitive.microsoft.com/text/analytics/v2.0/"

# The service endpoint of the language detection API for your region
language_api_url = text_analytics_base_url + "languages"
print(language_api_url)

# The payload to the API consists of a list of documents, each of which in turn contains
# an id and a text attribute. The text attribute stores the text to be analyzed.
documents = { 'documents': [
    { 'id': '1', 'text': 'This is a document written in English.' },
    { 'id': '2', 'text': 'Este es un document escrito en Español.' },
    { 'id': '3', 'text': '这是一个用中文写的文件' }
]}

# The next few lines of code call out to the language detection API using
# the requests library in Python to determine the language in the documents.
import requests
from pprint import pprint
headers   = {"Ocp-Apim-Subscription-Key": subscription_key}
response  = requests.post(language_api_url, headers=headers, json=documents)
pprint(response)
languages = response.json() # convert to json format
pprint(languages)

# The following lines of code render the JSON data as an HTML table.
from IPython.display import HTML
table = []
for document in languages["documents"]:
    text  = next(filter(lambda d: d["id"] == document["id"], documents["documents"]))["text"]
    langs = ", ".join(["{0}({1})".format(lang["name"], lang["score"]) for lang in document["detectedLanguages"]])
    table.append("<tr><td>{0}</td><td>{1}</td>".format(text, langs))
HTML("<table><tr><th>Text</th><th>Detected languages(scores)</th></tr>{0}</table>".format("\n".join(table)))

print("\n2. Analyze sentiment ")
sentiment_api_url = text_analytics_base_url + "sentiment"
print(sentiment_api_url)

# Each document is a tuple consisting of the id, the text to be analyzed and 
# the language of the text. You can use the language detection API 
# from the previous section to populate this field.
documents = {'documents' : [
  {'id': '1', 'language': 'en', 'text': 'I had a wonderful experience! The rooms were wonderful and the staff was helpful.'},
  {'id': '2', 'language': 'en', 'text': 'I had a terrible time at the hotel. The staff was rude and the food was awful.'},  
  {'id': '3', 'language': 'es', 'text': 'Los caminos que llevan hasta Monte Rainier son espectaculares y hermosos.'},  
  {'id': '4', 'language': 'es', 'text': 'La carretera estaba atascada. Había mucho tráfico el día de ayer.'}
]}

# The sentiment API can now be used to analyze the documents for their sentiments.
headers   = {"Ocp-Apim-Subscription-Key": subscription_key}
response  = requests.post(sentiment_api_url, headers=headers, json=documents)
sentiments = response.json()
pprint(sentiments)

print("\n3. Extract key phrases")
key_phrase_api_url = text_analytics_base_url + "keyPhrases"
print(key_phrase_api_url)

pprint(documents)

headers   = {"Ocp-Apim-Subscription-Key": subscription_key}
response  = requests.post(key_phrase_api_url, headers=headers, json=documents)
key_phrases = response.json()
pprint(key_phrases)

from IPython.display import HTML
table = []
for document in key_phrases["documents"]:
    text    = next(filter(lambda d: d["id"] == document["id"], documents["documents"]))["text"]    
    phrases = ",".join(document["keyPhrases"])
    table.append("<tr><td>{0}</td><td>{1}</td>".format(text, phrases))
HTML("<table><tr><th>Text</th><th>Key phrases</th></tr>{0}</table>".format("\n".join(table)))