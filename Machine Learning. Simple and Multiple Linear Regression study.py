#!/usr/bin/env python
# coding: utf-8

# Machine learning algorithms
# Data set is of health survey of 400 people, some with sleep disorders some without
# [HERE](Https://www.bbc.co.uk)

# Imports

# In[1]:


import pandas as pd
import opendatasets as od
import numpy as np
import sklearn.metrics as metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model,preprocessing,svm,preprocessing
from sklearn.metrics import f1_score,jaccard_score,log_loss,confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})



df = pd.read_csv('sleep-health-and-lifestyle-dataset\Sleep_health_and_lifestyle_dataset.csv')
df.head()


# Process the DataFrame to transform categorical values into integers, and remove occupation and blood pressure columns

# In[4]:


df_processed = df.drop(columns=["Occupation","Blood Pressure"],axis=1)
df_processed["Sleep Disorder"].replace(['None', 'Insomnia','Sleep Apnea'], [0,1,2], inplace=True)
df_processed["Gender"].replace(['Male', 'Female'], [0,1], inplace=True)
df_processed["BMI Category"].replace(['Normal', 'Normal Weight', 'Obese', 'Overweight'], [0,0,1,2], inplace=True)

df_processed.head()


# In[7]:


histogram = df_processed[['Physical Activity Level','Gender','Age','Sleep Duration','Quality of Sleep', 'Stress Level', 'BMI Category', 'Heart Rate','Daily Steps', 'Sleep Disorder']]
histogram.hist(figsize=(12,10),grid=False)
plt.tight_layout()
plt.show()


# SIMPLE LINEAR REGRESSION

# Define the dependent and independent variables to be tested.
# Here I'm looking to see if there is a correlation between Age and Quailty of Sleep.

# In[591]:


dependent = 'Heart Rate' # Y
independent = 'Sleep Duration' # X

print(df_processed[[dependent, independent]].head())


# Plot the dataset

# In[4]:


plt.figure(figsize=(8,6))
plt.scatter(df_processed[independent],df_processed[dependent])
plt.grid(True)
plt.ylabel("%s" % dependent)
plt.xlabel("%s" % independent)
plt.show()


# Select a random set of rows to be used as the training set, the remaining will be used to test the model.
# Create a list of random numbers, and make 80% of them 'True' to create an 80:20 split of train to test data at random.
# 
# Define new arrays to hold the dependent and independent variable data sets for both the training and testing sets.

# In[593]:


msk = np.random.rand(len(df_processed)) < 0.8
train = df_processed[msk]
test = df_processed[~msk]

train_y = np.asanyarray(train[[dependent]])
train_x = np.asanyarray(train[[independent]])

test_y = np.asanyarray(test[[dependent]])
test_x = np.asanyarray(test[[independent]])


# Use the SKLearn module to fit the linear regression model. After training the model using .fit method on the testing set, the .predict method will calculate the predicted slope and y-intercept of the model. The resulting line will then be used to compare the model to the testing set later.

# In[594]:


regr = linear_model.LinearRegression()
regr.fit(train_x,train_y)
yhat= regr.predict(test_x)


# Print the slope and intercept, respectively

# In[595]:


# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# Compare the resulting model to the training set

# In[596]:


plt.figure(figsize=(8,6))
plt.scatter(train_x,train_y, color='blue',label='Training data')
plt.grid(True)
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r',label='Predicted Model')
plt.xlabel("%s"%independent)
plt.ylabel("%s"%dependent)
plt.legend(fontsize=10)
plt.show()


# In[597]:


print("Mean absolute error: %.2f" % np.mean(np.absolute(yhat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((yhat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , yhat))


# Given the high MAE, MSE and low R2 score (<0.5), we can assume that, while there is some correlation present between heart rate and sleep duration, it's strength is low.

# MULTIPLE LINEAR REGRESSION
# 
# Test if there is a correlation with resting Heart Rate and Sleep Duration, Daily Steps, and Physical Activity Level.

# Split the processed dataframe into a features set and a target set.
# 

# In[599]:


features = np.asarray(df_processed[["Sleep Duration","Daily Steps", "Physical Activity Level"]])
target = df_processed["Heart Rate"]


# Use the SKlearn module to automatically split the dataset into train and test sets.

# In[598]:


x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=10)

print("Number of test samples :", x_test.shape[0])
print("Number of training samples:",x_train.shape[0])


# Use SKLearn again to train the model and calculate predictions.

# In[584]:


LinearReg = LinearRegression()
LinearReg.fit(x_train, y_train)

predictions = LinearReg.predict(x_test)


# In[585]:


print("Mean absolute error: %.2f" % mean_absolute_error(y_test, predictions))
print("Mean squared error: %.2f" % mean_squared_error(y_test, predictions))
print("R2-score: %.2f" % r2_score(y_test, predictions))


# In[589]:


intercept = LinearReg.intercept_
coeffs = LinearReg.coef_

coefficients = pd.DataFrame(zip(data.columns, coeffs))
coefficients


# From the coefficients we can say that, for this data set, sleep duration had the highest correlation on heart rate over daily steps and physical activity level. Like before, the poor error scores show only a weak correlation. This could also be because the correlations are actually non-linear.
