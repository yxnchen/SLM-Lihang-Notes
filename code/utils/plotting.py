# -*- coding: utf-8 -*-
"""
Created on: 2019/4/10 13:31

@author: its_cyx
"""

import matplotlib
import matplotlib.pyplot as plt


def plot_digit(image):
    plt.imshow(image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()
