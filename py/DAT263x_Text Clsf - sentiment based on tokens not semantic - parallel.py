# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 14:03:12 2018

@author: T901
"""
import numpy as np
import pandas as pd
import dask.dataframe as dd
import os
import string
import nltk
from nltk.stem.porter import PorterStemmer

np.random.seed(1234)

#tweets = pd.read_csv("D:\\data_science\\python_ws\\IntroAI_DAT263x_edX\\DAT263x-demos\\Text and Speech\\tweets.csv")
#stopwords = pd.read_csv("D:\\data_science\\python_ws\\IntroAI_DAT263x_edX\\DAT263x-demos\\Text and Speech\\stopwords.csv")
tweets = pd.read_csv("./DAT263x-demos/Text and Speech/tweets.csv")
stopwords = pd.read_csv("./DAT263x-demos/Text and Speech/stopwords.csv")
#tweets = dd.read_csv("./DAT263x-demos/Text and Speech/tweets.csv")
#stopwords = dd.read_csv("./DAT263x-demos/Text and Speech/stopwords.csv")

print(type(tweets), type(stopwords))

#import multiprocessing

#pool = multiprocessing.Pool()
from dask.multiprocessing import get

print(""" \n1. Normalize text, remove stop words, and stem """)
def azureml_main(dataset, stop_words):
   
    ## Give the columns names and make a list of the tweets
    dataset.columns = ['sentiment', 'tweets']
    #tweets = dataset['tweets'].tolist()
    tweets = dataset['tweets']
    print(type(tweets))
    #tweets_ddf = dd.from_pandas(tweets, npartitions=4)
    
    ## For each tweet, remove punctuation and set to lower case 
    sp = string.punctuation
    #tweets = list(map(lambda t: ''.join(["" if c.isdigit() else c for c in t]), tweets))
    tweets_dg = dd.from_pandas(tweets, npartitions=4).\
                map_partitions(
                    lambda tweets: tweets.map(
                        lambda t: ''.join(["" if c.isdigit() else c for c in t]))).\
                    compute()
    #tweets = list(map(lambda t: ''.join(["" if c in sp else c for c in t]), tweets))
    tweets_sp = dd.from_pandas(tweets_dg, npartitions=4).\
                map_partitions(
                    lambda tweets_dg: tweets_dg.map(
                        lambda t: ''.join(["" if c in sp else c for c in t]))).\
                    compute()          
    #tweets = list(map(str.lower, tweets))
    tweets_lc = dd.from_pandas(tweets_sp, npartitions=4).\
                map_partitions(
                    lambda tweets_sp: tweets_sp.map(
                        lambda t: t.lower())).\
                    compute()
    # stem the tweet text
    porter_stemmer = PorterStemmer()
    temp = [tweet.split() for tweet in tweets] ## Split tweets into tokens
    #tweets = list(map(lambda t: ' '.join([porter_stemmer.stem(word) for word in t.split()]), tweets))
    tweets_st = dd.from_pandas(tweets_lc, npartitions=4).\
                map_partitions(
                    lambda tweets_lc: tweets_lc.map(
                        lambda t: ' '.join([porter_stemmer.stem(word) for word in t.split()]))).\
                    compute()
    # Remove stop words
    stop_words = [w for w in stop_words.words if w in stop_words.words.unique() ]
    #tweets = list(map(lambda t: ' '.join([word for word in t.split() if word not in set(stop_words)]), tweets))
    tweets_sw = dd.from_pandas(tweets_st, npartitions=4).\
                map_partitions(
                    lambda tweets_st: tweets_st.map(
                        lambda t: ' '.join([word for word in t.split() if word not in set(stop_words)]))).\
                    compute()
    
     ## Set the sentiment values to -1 and 1
    #dataset['sentiment'] = [1 if s == 4 else -1 for s in dataset['sentiment']]
    dataset['sentiment'] = [1 if s == 4 else 0 for s in dataset['sentiment']]
    
    dataset['tweets'] = tweets_sw
    return dataset

import time

start = time.time()
tweets_ds = azureml_main(tweets, stopwords)
end = time.time()

print(f'\nTime to complete step 1: {end -start: .2f}s')

print(""" \n2. Feature Hashing: Hash text tokens to numeric indicator features """)
""" See http://scikit-learn.org/stable/modules/feature_extraction.html#vectorizing-a-large-text-corpus-with-the-hashing-trick 
See https://docs.microsoft.com/en-us/azure/machine-learning/studio-module-reference/feature-hashing
The hash function employed is the signed 32-bit version of Murmurhash3
"""
from sklearn.feature_extraction.text import HashingVectorizer, FeatureHasher

start = time.time()
hv = HashingVectorizer(ngram_range = (1, 2), analyzer='word', norm='l2', n_features = 32768)  # 2^15
tweets_hash = hv.transform(tweets_ds['tweets'])
end = time.time()

print(f'\nTime to complete step 2: {end -start: .2f}s')
#hv = FeatureHasher(n_features = 32768)  # 2^15
#tweets_hash = [hv.transform(s) for s in tweets_ds['tweets']] 
print(tweets_hash.shape, type(tweets_hash))
print(type(tweets_hash[0, :]), tweets_hash[0, :].shape)
print(tweets_hash[0, :].min(), tweets_hash[0, :].max(), len(np.unique(tweets_hash[0, :])))

print(""" \n3. Split the data for training and testing """)
from sklearn.model_selection import train_test_split

#data = pd.DataFrame(tweets_hash)
#data['sentiment'] = tweets_ds['sentiment']

"""
Use the same random_state to have the matching indices in input and output data split
Try to fix random state by declaring np.random.seed(1234 but resulted AUC is low ~ 0.5
"""
#train_data, test_data = train_test_split(tweets_hash, test_size = 0.3, random_state = 1234)
#y_train, y_test = train_test_split(tweets_ds['sentiment'], test_size = 0.3, random_state = 1234)
start = time.time()
train_data, test_data, y_train, y_test = train_test_split(tweets_hash, tweets_ds['sentiment'], test_size = 0.3, random_state = 1234)
end = time.time()

print(f'\nTime to complete step 3: {end -start: .2f}s')
#train_data, test_data = train_test_split(tweets_hash, test_size = 0.3)
#y_train, y_test = train_test_split(tweets_ds['sentiment'], test_size = 0.3)

#train_data = train[:, :-1]
#y_train = train[:, -1]

#test_data = test[:, :-1]
#y_test = test[:, -1]

print(train_data.shape, y_train.shape, test_data.shape, y_test.shape)

print(""" \n4. Train the model using two-class Boosted Decision Tree Classifier """)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import dask_ml.joblib  # registers joblib plugin
# Scikit-learn bundles joblib, so you need to import from
# `sklearn.externals.joblib` instead of `joblib` directly
from sklearn.externals.joblib import parallel_backend

start = time.time()
# import dask_ml.joblib   # registers joblib plugin
# Scikit-learn bundles joblib internally, so you need to import from
# `sklearn.externals.joblib` instead of `joblib` directly

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1),
                         algorithm="SAMME.R",
                         n_estimators=200)

with parallel_backend('threading'):
    # Your normal scikit-learn code here
   
    # Train the model using training set
    bdt.fit(train_data, y_train)
    print("Done fit")
    
# Make prediction using testing set
y_pred = bdt.predict(test_data)

end = time.time()

print(f'\nTime to complete step 4: {end -start: .2f}s')

print(""" \n5. Classification metrics """)
# See http://scikit-learn.org/stable/modules/classes.html
# See http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
from sklearn.metrics import accuracy_score, precision_score, \
                            recall_score, f1_score, roc_auc_score, roc_curve

start = time.time()
print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))
print("Precision: %.2f" % precision_score(y_test, y_pred))
print("Recall: %.2f" % recall_score(y_test, y_pred))
print("F1_score: %.2f" % f1_score(y_test, y_pred))
# Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
#print("AUC: %.2f" % roc_auc_score(y_test, y_pred))
probas_ = bdt.predict_proba(test_data)[:, 1]
print("AUC: %.2F" % roc_auc_score(y_test, probas_))

# Compute Receiver operating characteristic (ROC)
fpr, tpr, _ = roc_curve(y_test, probas_)
end = time.time()

print(f'\nTime to complete step 5: {end -start: .2f}s')

import matplotlib.pyplot as plt
plt.figure()
lw = 2 # linewidth
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, probas_))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show() 

print(""" 6. Print results """)
"""
test_data is a scipy sparse matrix, use toarray() or todense() to produce a
numpy matrix, which does work for DataFrame constructor (pandas df constructor)
See: https://stackoverflow.com/questions/36967666/transform-scipy-sparse-csr-to-pandas?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

result_ds = pd.DataFrame(test_data.toarray()) >>> causing memory error
"""

"""
See: https://stackoverflow.com/questions/17241004/pandas-how-to-get-the-data-frame-index-as-an-array?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
See: https://stackoverflow.com/questions/43772362/how-to-print-a-specific-row-of-a-pandas-dataframe?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

result_ds = tweets_ds['tweets'].loc[[y_test.index] >>> error: unhashable type: 'Int64Index' since y_test.index is Int64Index
loc: selection by label
"""
result_ds = tweets_ds['tweets'].loc[y_test.index.tolist()]
result_ds['sentiment_label'] = y_test
result_ds['sentiment_pred'] = y_pred
result_ds['probability'] = probas_
print(result_ds)
