#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 18:36:53 2019

@author: rohitmadan
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors, linear_model
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
plt.style.use('ggplot')

df1 = pd.read_csv('/Users/rohitmadan/Desktop/breastcan.csv')
#df1.set_index(df1.iloc[0], drop= False)
X=df1.drop(labels='Class',axis=1)
#print(X)
y=df1['Class']
print(y)

#y = y1 < 3  # is the rating <= 3?
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
knn = neighbors.KNeighborsClassifier(n_neighbors = 5)
knn_model_1 = knn.fit(X_train, y_train)
#print('k-NN accuracy for test set: %f' % knn_model_1.score(X_test, y_test))


y_true, y_pred = y_test, knn_model_1.predict(X_test)
print(classification_report(y_true, y_pred))