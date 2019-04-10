# -*- coding: utf-8 -*-
"""
Created on: 2019/4/10 11:28

@author: its_cyx
"""

from utils import load_mldata, plotting

# 加载 MNIST 数据集
# 有70000张图片，大小为28*28，因此共784个特征，每个像素的值在0~255之间
# 训练集 60000，测试集 10000
X_train, y_train, X_test, y_test = load_mldata.fetch_mnist()

# 画图像
x = 258
some_x = X_train[x,:]
some_x_image = some_x.reshape(28,28)
print(y_train[x])
plotting.plot_digit(some_x_image)

