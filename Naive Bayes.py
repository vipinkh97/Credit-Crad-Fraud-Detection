# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 17:10:04 2019

# Credit Card Fraud Detection
# Naive Bayes

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Evaluating the Confusion Matrix
true_pos = cm[1][1]
false_pos = cm[1][0]
false_neg = cm[0][1]
precision = (true_pos /(true_pos+false_pos))
recall = (true_pos /(true_pos + false_neg))
f1_score = 2*(precision*recall)/(precision+recall)


"""
Precision: 0.8514
Recall: 0.0659
F1-Score: 0.1223     

"""