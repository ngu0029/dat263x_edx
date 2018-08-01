# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 13:47:36 2018

@author: T901
"""

# install textblob library and define functions for TF-IDF
#!pip install -U textblob
import math
from textblob import TextBlob as tb
"""
https://pypi.python.org/pypi/textblob
TextBlob is a Python (2 and 3) library for processing textual data. 
It provides a simple API for diving into common natural language processing (NLP) tasks 
such as part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, 
translation, and more.
"""
# Get standard stop words from NLTK
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from string import punctuation
import urllib.request

# Get a first document, normalize it, and remove stop words
urllib.request.urlretrieve('https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/Moon.txt', 'Moon.txt')
doc1 = open("Moon.txt", "r")
doc1Txt = doc1.read()
txt = ''.join(c for c in doc1Txt if not c.isdigit())
txt = ''.join(c for c in txt if c not in punctuation).lower()
txt = ' '.join([word for word in txt.split() if word not in (stopwords.words('english'))])
doc1.close()

# Get a second document, normalize it, and remove stop words
urllib.request.urlretrieve('https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/Gettysburg.txt', 'Gettysburg.txt')
doc2 = open("Gettysburg.txt", "r")
doc2Txt = doc2.read()
#print (doc2Txt)
txt2 = ''.join(c for c in doc2Txt if not c.isdigit())
txt2 = ''.join(c for c in txt2 if c not in punctuation).lower()
txt2 = ' '.join([word for word in txt2.split() if word not in (stopwords.words('english'))])
doc2.close()

# and a third
print("------------------------------------------------")
doc3 = urllib.request.urlretrieve('https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/Cognitive.txt','Cognitive.txt')
doc3 = open("Cognitive.txt", "r")
doc3Txt = doc3.read()
#print (doc3Txt)
from string import punctuation
txt3 = ''.join(c for c in doc3Txt if not c.isdigit())
txt3 = ''.join(c for c in txt3 if c not in punctuation).lower()
txt3 = ' '.join([word for word in txt3.split() if word not in (stopwords.words('english'))])
doc3.close()

def tf(word, doc):
    return doc.words.count(word) / len(doc.words)

def contains(word, docs):
    return sum(1 for doc in docs if word in doc.words)

def idf(word, docs):
    return math.log(len(docs) / (1 + contains(word, docs)))

def tfidf(word, doc, docs):
    return tf(word,doc) * idf(word, docs)


# Create a collection of documents as textblobs
doc1 = tb(txt)
doc2 = tb(txt2)
doc3 = tb(txt3)
docs = [doc1, doc2, doc3]

# Use TF-IDF to get the three most important words from each document
print('-----------------------------------------------------------')
""" See https://docs.python.org/3/library/functions.html#enumerate 
https://stackoverflow.com/questions/2191699/find-an-element-in-a-list-of-tuples?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
returns a list of tuples!!!!
"""
for i, doc in enumerate(docs):
    print("Top words in document {}".format(i + 1))
    
    """ Create a dictionary of words with tfidf scores 
    See: https://developmentality.wordpress.com/2012/03/30/three-ways-of-creating-dictionaries-in-python/
    """
    scores = {word: tfidf(word, doc, docs) for word in doc.words}
    
    """ Sort the scores dictionary by its tfidf score in reverse order
    See: https://www.saltycrane.com/blog/2007/09/how-to-sort-python-dictionary-by-keys/
         https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
         
    scores.items: a list of dict items, in format [(,), (,),...]
    
    https://docs.python.org/3/howto/sorting.html
    Both list.sort() and sorted() have a key parameter to specify a function to be called on 
    each list element prior to making comparisons.     
    """
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:3]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))