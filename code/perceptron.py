# -*- coding: utf-8 -*-
"""
Created on: 2019/4/10 13:46

@author: its_cyx
"""

from utils import load_mldata
from sklearn.linear_model import Perceptron

X_train, y_train, X_test, y_test = load_mldata.fetch_mnist()

# 二分类：数字5和其他
y_train_5 = (y_train == 5)
y_train_5 = y_train_5.ravel()
y_test_5 = (y_test == 5)
y_test_5 = y_test_5.ravel()

classifier = Perceptron(tol=1e-3, random_state=2019)
classifier.fit(X_train, y_train_5)
classifier.score(X_test, y_test_5)
