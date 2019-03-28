# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 17:10:04 2019

# Credit Card Fraud Detection
# Random Forest

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

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0,n_jobs=4)
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
Trees = 10
Precision: 0.7623
Recall: 0.9058
F1-Score: 0.8279     

Trees = 50
Precision: 0.7821
Recall: 0.9186
F1-Score: 0.8449  

80:20(Trainingset: Testset)
Trees = 200
Precision: 0.7920
Recall: 0.9195
F1-Score: 0.8510  
 
70:30(Trainingset: Testset)
Trees = 200
Precision: 0.7687
Recall: 0.94166
F1-Score: 0.84644      


"""