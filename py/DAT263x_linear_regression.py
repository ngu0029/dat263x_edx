# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 14:17:42 2018

@author: T901
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
np.random.seed(1)                              # for fixing random values

exercise = pd.read_csv("D:\\data_science\\python_ws\\IntroAI_DAT263x_edX\\DAT263x-demos\\ML\\exercise.csv")
calories = pd.read_csv("D:\\data_science\\python_ws\\IntroAI_DAT263x_edX\\DAT263x-demos\\ML\\calories.csv")

text1 = """ 1. Join two datasets """
print(text1)
# exercise and calories dataframes have the same 'User_ID' field
frame = exercise.merge(calories, on = 'User_ID', how = 'left') # Left outer join

print(frame.dtypes)

# Create a scatter plot matrix
#%matplotlib inline # only for jupyter notebook
print("Create a scatter plot matrix")
num_cols = ["Age", "Height", "Weight", "Duration",
            "Heart_Rate", "Body_Temp", "Calories"]
sns.pairplot(frame[num_cols], size=2)

text2 = """ 2. Make gender categorical """
print(text2)
# Gender field is string type (object), converting to categorical type
frame['Gender'] = frame['Gender'].astype('category')

print("sklearn requires all inputs to be numeric, we should convert all our categorical variables \
      into numeric by encoding the categories")

le = LabelEncoder()
frame['Gender'] = le.fit_transform(frame['Gender'])

"""
See: https://stackoverflow.com/questions/32011359/convert-categorical-data-in-pandas-dataframe?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
"""
#frame.Gender = frame.Gender.apply(lambda x: x.cat.codes)
"""
See: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

"""
#from sklearn.preprocessing import label_binarize
# Binarize the output
#y = label_binarize(y, classes=[0, 1, 2])

text3 = """ 3. Create features for Duration and Heart_Rate^2 """
print(text3)
""" Duration vs Calories / Heart_Rate vs Calories have curved relationship (not exactly linear - noted that 
we'll build regression model) in scatter plot.Squaring these features is to give better relationship to Calories """
frame['Duration_sqr'] = frame['Duration'].pow(2)
frame['Heart_Rate_sqr'] = frame['Heart_Rate'].pow(2)

text4 = """ 4. Calculate log Calories to ensure no negative predictions """
print(text4)
""" 
So we're converting that to a log so that our prediction will actually be log of calories, rather than calories, and
it will always be a positive number.
"""
# Replace Calories field by its natural logarithm
frame['Calories_ln'] = np.log(frame.Calories)

text5 = """ 5. Scale numerics with 'normal' distribution using z-score"""
"""See https://stackoverflow.com/questions/24761998/pandas-compute-z-score-for-all-columns?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
"""
print(text5)
cols = ["Height", "Weight", "Heart_Rate", "Body_Temp", "Heart_Rate_sqr"]
for col in cols:
        frame[col] = (frame[col] - frame[col].mean())/frame[col].std(ddof=0)
        
text6 = """ 6. Scale non normal columns using MinMax """
"""See https://machinelearningcoban.com/general/2017/02/06/featureengineering/#rescaling

or This is how you can do it using sklearn and the preprocessing module. Sci-Kit Learn has many pre-processing functions 
for scaling and centering data.
See https://stackoverflow.com/questions/21764475/scaling-numbers-column-by-column-with-pandas-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
"""
print(text6)
cols = ["Age", "Duration", "Duration_sqr"]
for col in cols:
        frame[col] = (frame[col] - frame[col].min())/(frame[col].max() - frame[col].min())

text7 = """ 7. User_ID and Calories aren't predictive features """
""" 
Well, user ID is not really a very useful predictive column, it's just a unique ID for each user,
it shouldn't have any bearing on how many calories they burn.
And calories itself is the original label that I'm trying to predict,
although now I'm trying to predict the log of calories. So I don't wanna use calories as a feature, 
and so I'm basically just saying, these aren't features, 
don't consider them when we're building the model. 
"""
print(text7)   

text8 = """ 8. Randomly split into 70% training and 30% test """
"""
See: https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
"""
print(text8)

train, test = train_test_split(frame, test_size=0.3)

text9 = """ 9. Train Linear Regression model """
""" See http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html """

print(text9)

linreg = linear_model.LinearRegression()   # make an instance of LinearRegression object

train_data = train[["Gender", "Age", "Height", "Weight", "Duration",
                   "Heart_Rate", "Body_Temp", "Duration_sqr", "Heart_Rate_sqr"]]
train_label = train["Calories_ln"]

test_data = test[["Gender", "Age", "Height", "Weight", "Duration",
                   "Heart_Rate", "Body_Temp", "Duration_sqr", "Heart_Rate_sqr"]]
test_label = test["Calories_ln"]

# Train the model using the training set
linreg.fit(train_data, train_label)

# Make prediction using the testing set
y_pred = linreg.predict(test_data)

text10 = "10. Get model evaluation metrics"
print(text10)
# The coefficients
print("Coefficients: \n", linreg.coef_)
print("Mean square error: %.2f" % mean_squared_error(test_label, y_pred))
# Explained variance score: 1 is perfect prediction
print("Variance score: %.2f" % r2_score(test_label, y_pred))

text11 = "11. Print final dataframe"
print(text11)     
test["Calories_ln_pred"] = y_pred
test["Calories_pred"] = np.exp(y_pred)
print(test)
        
        