# -*- coding: utf-8 -*-
"""
Created on: 2019/4/10 13:25

@author: its_cyx
"""

import numpy as np
from scipy.io import loadmat


def fetch_mnist():
    mnist = loadmat('dataset/mnist-original.mat')

    # MNIST 有70000张图片，大小为28*28，因此共784个特征
    # 每个像素的值在0~255之间
    X, y = mnist["data"].transpose(), mnist["label"].transpose()
    print(X.shape)
    print(y.shape)

    shuffle_idx = np.random.permutation(len(y))
    X_train, y_train, X_test, y_test = \
        X[shuffle_idx[:60000]], y[shuffle_idx[:60000]], X[shuffle_idx[60000:]], y[shuffle_idx[60000:]]
    return X_train, y_train, X_test, y_test
