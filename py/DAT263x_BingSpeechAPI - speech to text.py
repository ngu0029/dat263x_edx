# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 12:56:18 2018

@author: T901
"""
"""
Go to your Azure portal webpage > Choose All Resources > Choose YOUR Bing Speech API service
> Choose Keys to see the subscription key > copy it to the speechKey below
"""
speechKey = '443d193a670144c7b13609bdb50db3bc'

# Install SpeechRecognition package
#!pip install SpeechRecognition
#!pip install pyaudio
# See: https://stackoverflow.com/questions/12332975/installing-python-module-within-code?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
"""
import pip

def install(package):
    pip.main(['install', package])
    
install('SpeechRecognition')
install('pyaudio')    
"""

# Convert Speech to Text
import speech_recognition as sr

# Read the audio file
r = sr.Recognizer()  # Creates a new Recognizer instance, which represents a collection of speech recognition functionality.
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)
"""
Definition : listen(source, timeout=None, phrase_time_limit=None, snowboy_configuration=None)
The timeout parameter is the maximum number of seconds that this will wait for a phrase 
to start before giving up and throwing an speech_recognition.WaitTimeoutError exception. 
If timeout is None, there will be no wait timeout.

The phrase_time_limit parameter is the maximum number of seconds that this will allow 
a phrase to continue before stopping and returning the part of the phrase processed 
before the time limit was reached. The resulting audio will be the phrase cut off at the time limit. 
If phrase_timeout is None, there will be no phrase time limit.

This operation will always complete within timeout + phrase_timeout seconds if both are numbers, 
either by returning the audio data, or by raising a speech_recognition.WaitTimeoutError exception.

"""    

# transcribe speech using the Bing Speech API
try:
    transcription = r.recognize_bing(audio, key=speechKey)   # Microsoft Bing Speech API
    print("Here's what I heard:")
    print('"' + transcription + '"')

except sr.UnknownValueError:
    print("The audio was unclear")
except sr.RequestError as e:
    print (e)
    print("Something went wrong :-(; {0}".format(e))
    
"""
Definition : recognize_bing(audio_data, key, language=en-US, show_all=False)

Returns the most likely transcription if show_all is false (the default). 
Otherwise, returns the raw API response as a JSON dictionary.

Raises a speech_recognition.UnknownValueError exception if the speech is unintelligible. 
Raises a speech_recognition.RequestError exception if the speech recognition operation failed, 
if the key isn't valid, or if there is no internet connection.
"""