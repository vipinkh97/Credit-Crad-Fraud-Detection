"""
Created on Mon Feb 11 15:47:17 2019

# Credit Card Fraud Detection
# Logistic Regression

@author: Vipin Dhonkaria
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('creditcard.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 30].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# y_score = classifier.decision_function(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Evaluating the Confusion Matrix
true_pos = cm[1][1]
false_pos = cm[1][0]
false_neg = cm[0][1]
true_neg = cm[0][0]
precision = (true_pos /(true_pos+false_pos))
recall = (true_pos /(true_pos + false_neg))
sensitivity = (true_pos/(true_pos + false_neg))
specificity = (true_neg/(true_neg+ false_pos))
f1_score = 2*(precision*recall)/(precision+recall) 

#from sklearn.metrics import average_precision_score
#average_precision = average_precision_score(y_test, y_score) 

"""
80:20(Training set: Test set)
Precision: 0.6336
Recall: 0.8767
F1-Score: 0.7356   

70:30(Training set: Test set)
Precision: 0.6190
Recall: 0.8834
F1-Score: 0.728    

"""
