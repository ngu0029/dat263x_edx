# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 10:37:54 2018

@author: T901
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, \
                            recall_score, f1_score, roc_auc_score, auc, roc_curve

np.random.seed(1234)                              # for fixing random values

# Fix: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe1 in position 11: invalid continuation byte
# See https://stackoverflow.com/questions/19699367/unicodedecodeerror-utf-8-codec-cant-decode-byte
diabetes = pd.read_csv("D:\\data_science\\python_ws\\IntroAI_DAT263x_edX\\DAT263x\\Lab01\\diabetes.csv", encoding="ISO-8859-1")
doctors = pd.read_csv("D:\\data_science\\python_ws\\IntroAI_DAT263x_edX\\DAT263x\\Lab01\\doctors.csv", encoding="ISO-8859-1")

text1 = """ 1. Join two datasets """
print(text1)
# diabetes and doctors dataframes have the same 'User_ID' field
frame = diabetes.merge(doctors, on = 'PatientID', how = 'left') # Left outer join

# Create a scatter plot matrix
#%matplotlib inline

num_cols = ["PlasmaGlucose", "DiastolicBloodPressure", "TricepsThickness", "SerumInsulin", "BMI",
                   "Pregnancies", "DiabetesPedigree", "Age"]
sns.pairplot(frame[num_cols], size=2)

text2 = """ 2. Log Age for more linear relationships """
""" Age is skewed towards younger patients, so use log for more linear relationships """
print(text2)
frame['Age_ln'] = np.log(frame['Age'])

text3 = """ 3. Scale numeric with 'normal' distribution using Z-Score """
print(text3)
cols = ["PlasmaGlucose", "DiastolicBloodPressure", "TricepsThickness", "BMI"]
for col in cols:
    frame[col] = (frame[col] - frame[col].mean())/frame[col].std(ddof=0)
    
text4 = "4. Scale non normal distributions with MinMax"
print(text4)
cols = ["SerumInsulin", "Pregnancies", "DiabetesPedigree", "Age", "Age_ln"]
for col in cols:
    frame[col] = (frame[col] - frame[col].min())/(frame[col].max() - frame[col].min())

text5 = "5. PatientID and Physician aren't predictive features"
print(text5)

text6 = """6. Randomly split into 70% training and 30% test """
"""
See: https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
"""
print(text6)
train, test = train_test_split(frame, test_size=0.3)   

text7 = """ 7. Train Two class Boosted Decision Tree model """
""" See http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html 
See https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/
"""
print(text7)
# Create and fit an AdaBoosted decision tree
""" AdaBoostClassifier: Algorithm SAMME.R shows better performance than SAMME.
Decision Tree Classifier support calculation of class probabilities.

max_depth tradeoffs ROC performance; max_depth goes high, ROC performance decreases
but Scored_Proba becomes more discriminative btw class 0 vs. class 1 (try max_depth = None, 10, 100)

When max_depth not specified, BaggingClassifier outperforms AdaBoostClassifier
When max_depth = 1, AdaBoostClassifier (AUC = 0.95) outperforms BaggingClassifier, buts its Score_Proba is NOT discriminative
When max_depth = 10, AdaBoostClassifier (AUC = 0.94) outperforms BaggingClassifier, buts its Score_Proba is GOOD discriminative

Not trying learning_rate = 0.2 (default = 1) for AdaBoostClassifier
Not trying different number of estimators (trees)

AdaBoostClassifier OUTPERFORMS!!!!!!!

Use the random seed as declared at the top of the program.
"""
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1),
#bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=100),
#bdt = BaggingClassifier(DecisionTreeClassifier(max_depth = 10),
                         algorithm="SAMME.R",
                         n_estimators=200)             # make an instance of AdaBoostClassifier object

train_data = train[["PlasmaGlucose", "DiastolicBloodPressure", "TricepsThickness", "SerumInsulin", "BMI",
                   "Pregnancies", "DiabetesPedigree", "Age", "Age_ln"]]
train_label = train["Diabetic"]

test_data = test[["PlasmaGlucose", "DiastolicBloodPressure", "TricepsThickness", "SerumInsulin", "BMI",
                   "Pregnancies", "DiabetesPedigree", "Age", "Age_ln"]]
test_label = test["Diabetic"]

# Train the model using the training set
bdt.fit(train_data, train_label)

# Make prediction using the testing set
y_pred = bdt.predict(test_data)

# Regression metrics
print("Mean square error: %.2f" % mean_squared_error(test_label, y_pred))
# Explained variance score: 1 is perfect prediction
print("Variance score: %.2f" % r2_score(test_label, y_pred))

# Classification metrics
# See http://scikit-learn.org/stable/modules/classes.html
# See http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
print("Accuracy: %.2f" % accuracy_score(test_label, y_pred))
print("Precision: %.2f" % precision_score(test_label, y_pred))
print("Recall: %.2f" % recall_score(test_label, y_pred))
print("F1_score: %.2f" % f1_score(test_label, y_pred))
# Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
#print("AUC: %.2f" % roc_auc_score(test_label, y_pred))
""" See https://stackoverflow.com/questions/30051284/plotting-a-roc-curve-in-scikit-yields-only-3-points?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
See http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
metrics.roc_auc_score(y_true, y_score[, …])
y_score: probability estimates, not y_pred
"""
probas_ = bdt.predict_proba(test_data)[:,1]
print("AUC: %.2f" % roc_auc_score(test_label, probas_))

# Compute Receiver operating characteristic (ROC)
fpr, tpr, _ = roc_curve(test_label, probas_, drop_intermediate=False)

import matplotlib.pyplot as plt
plt.figure()
lw = 2 # linewidth
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_score(test_label, probas_))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show() 

text8 = "8. Print final dataframe"
print(text8)
test['Diabetic_pred'] = y_pred
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier
cls_proba = bdt.predict_proba(test_data)
print(cls_proba, type(cls_proba), cls_proba.shape)
test['Scored_Proba'] = cls_proba[:, 1]
print(test)