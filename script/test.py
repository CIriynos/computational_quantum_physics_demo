#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

data = [
    [0.0, 0.0, 0],
    [0.2, 0.0, 0.1],
    [0.4, 0.0, 0.2],
    [0.6, 0.0, 0.3],
    [0.8, 0.0, 0.4],
    [0.0, 0.5, 0.5],
    [0.2, 0.5, 0.6],
    [0.4, 0.5, 0.7],
    [0.6, 0.5, 0.8],
    [0.8, 0.5, 0.9],
    [0.0, 1.0, 1.0],
    [0.2, 1.0, 1.1],
    [0.4, 1.0, 1.2],
    [0.6, 1.0, 1.3],
    [0.8, 1.0, 1.4],
]

data_sheet = np.array(data)
print(data_sheet)

X = np.reshape(data_sheet[:, 0], (3, 5))
print(X, "\n\n")

Y = np.reshape(data_sheet[:, 1], (3, 5))
print(Y, "\n\n")

Z = np.reshape(data_sheet[:, 2], (3, 5))
print(Z, "\n\n")

XX, YY = np.meshgrid(np.linspace(0, 0.8, 5), np.linspace(0, 1, 3))
print(X, "\n\n")
print(Y, "\n\n")

fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
ax.contourf(X, Y, Z)
plt.savefig("tmp")