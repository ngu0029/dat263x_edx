# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 20:00:40 2018

@author: T901
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score

#bl_donate = pd.read_csv("D:\\github\\dat263x_edx\\DAT263x-demos\\ML\\Blood donation data.csv")
bl_donate = pd.read_csv("./DAT263x-demos/ML/Blood donation data.csv")
bl_donate_orig = bl_donate.copy()
print(bl_donate.columns)
# Index(['Recency', ' Frequency', ' Monetary', ' Time', ' Class'], dtype='object')
# See there is a space before ' Frequency', ' Monetary', ' Time', ' Class'

# Select columns for clustering
cols = ["Recency", " Frequency", " Monetary", " Time"]

# see how data distribution looks
for c in cols:
    print(max(bl_donate[c]))
    
import seaborn as sns

sns.pairplot(bl_donate[cols], size = 2)

# Normalize data
for col in cols:
    bl_donate[col] = (bl_donate[col] - bl_donate[col].mean())/bl_donate[col].std(ddof=0)

# Train the model    
K = 3
X = bl_donate[cols]

print("\nTry k-means algorithm")
kmeans = KMeans(n_clusters = K).fit(X)

y_pred = kmeans.predict(X)    

# Metrics for regression
#print("Mean squared error: %.2f " % mean_squared_error(bl_donate[' Class'], y_pred))
# this metric R2 is only applied for linear regression, not clustering
#print("Variance score: %.2f " % r2_score(bl_donate[' Class'], y_pred))

# for classification
from sklearn.metrics import accuracy_score
print("Accuracy: %.2f" % accuracy_score(bl_donate[' Class'], y_pred))
"""Very low accuracy - trying different classification model"""

print("\nTry Softmax regression")
from sklearn.model_selection import train_test_split
train, test = train_test_split(bl_donate, test_size=0.3)
trainX, trainY, testX, testY = train[cols], train[' Class'], test[cols], test[' Class']
print(len(trainX), len(trainY), len(testX), len(testY))

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
logreg.fit(trainX, trainY)
predY = logreg.predict(testX)

print('Accuracy: %.2f ' % accuracy_score(testY, predY))

print("\nTry SVC")
from sklearn.svm import SVC
clf = SVC(kernel='rbf', degree=3, C=1e5)
clf.fit(trainX, trainY)
predY = clf.predict(testX)

print('Accuracy: %.2f ' % accuracy_score(testY, predY))

# For two dimensions (datasets with two columns)
def kmeans_display_2D(X, label):
    K = np.amax(label) + 1
    #print(K)
    X0 = X[label == 0, :]
    #print(X0)
    #print(X0.shape)
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)
    
    plt.axis('equal')
    plt.plot()
    plt.show()

# For 3D datasets
# See See http://www.bigendiandata.com/2017-04-18-Jupyter_Customer360/
def kmeans_display_3D(X, label):
    # Edict here
    return 0

# For N dimensions
# See http://www.bigendiandata.com/2017-04-18-Jupyter_Customer360/
"""
What if you’re clustering over more than 3 columns? How do you visualize that? 
One common approach is to split the 4th dimension data into groups 
and plot a 3D graph for each of those groups. 

Another approach is to split all the data into groups based on 
the k-means cluster value, then apply an aggregation function 
such as sum or average to all the dimensions in that group, 
then plot those aggregate values in a heatmap.

OR USE PCA METHOD (Dimensionality Reduction Algorithms) AS INTRODUCED IN THE LESSON OF THE COURSE
edX Home Page Microsoft: DAT263x
Introduction to Artificial Intelligence (AI)
"""
def kmeans_display_ND(X, label):
    # Edit here
    return 0

# create df with input columns
bl_donate_orig = bl_donate_orig[cols]

# normalize data using MinMax
for col in cols:
    bl_donate_orig[col] = (bl_donate_orig[col] - bl_donate_orig[col].min()) / (bl_donate_orig[col].max() - bl_donate_orig[col].min())
    
# ensure all values are positive (this is needed for our customer 360 use-case)
bl_donate_orig = bl_donate_orig.abs()    # Not neccesary, data already positive

# and add the Y
bl_donate_orig['Class_pred'] = y_pred

# split df into cluster groups
grouped = bl_donate_orig.groupby(['Class_pred'], sort=True)

# compute sums for every column in every group
#sums = grouped.sum()  # pivot df
means = grouped.mean()  # pivot df

print(means)

"""
Use seaborn: https://seaborn.pydata.org/generated/seaborn.heatmap.html
Or matplotlib: https://plot.ly/matplotlib/heatmaps/
"""
import seaborn as sns
#ax = sns.heatmap(sums, annot=True, fmt=".2f")
import matplotlib.pyplot as plt
plt.figure()
ax = sns.heatmap(means, annot=True, fmt=".2f")

"""
K = 2: Two clusters class_0 and class_1 show discriminative for the inputs Freq, Monetary, Time
K = 3: Cluster class_1 and Cluster class_2 (changable) look NOT discriminative
"""