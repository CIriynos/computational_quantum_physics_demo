#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import re

f1 = open("e1.txt", 'r', encoding="utf-8")
f2 = open("e2.txt", 'r', encoding="utf-8")
pattern = r'[-+]?\d+\.\d+'
data_sheet = np.zeros((100000, 2))

cnt1 = 0
while True:
    str1 = f1.readline()
    if str1 == "":
        break
    result1 = re.search(pattern, str1).group()
    data_sheet[cnt1, 0] = float(result1)
    cnt1 += 1

cnt2 = 0
while True:
    str2 = f2.readline()
    if str2 == "":
        break
    result2 = re.search(pattern, str2)
    if result2 == None:
        break
    data_sheet[cnt2, 1] = float(result2.group())
    cnt2 += 1

print(cnt1)
print(cnt2)

# plot
fig, ax = plt.subplots()
l1, = ax.plot(np.linspace(1, cnt1, cnt1), data_sheet[0:cnt1, 0], linewidth=2.0)
l2, = ax.plot(np.linspace(1, cnt1, cnt1), data_sheet[0:cnt1, 1], linewidth=2.0)

ax.legend(handles=[l1, l2,], labels=['e1', 'e2'])
plt.savefig("energy_cp_result")

f1.close()
f2.close()