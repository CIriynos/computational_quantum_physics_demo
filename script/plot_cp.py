#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

XYZ = 0
POLAR = 1
SPHERICAL = 2
POLAR_HALF = 3

data_name_list = sys.argv[1:]
file_dir = os.path.dirname(os.path.realpath('__file__'))
file_obj_list = []

for data_name in data_name_list:
    data_path = os.path.join(file_dir, "output/" + data_name + ".wavedata")
    f = open(data_path, 'r', encoding="utf-8")
    file_obj_list.append(f)

data_sheet_list = []

for k, f in enumerate(file_obj_list):
    meta_str = f.readline()
    meta_data = [int(x) for x in meta_str.split()[1:]]

    dimension = meta_data[0]
    count = meta_data[1]
    grid_mode = meta_data[2]

    if dimension != 1:
        print("sorry. It only supports 1d comparable plot.")
        quit()

    # the first line after metadata should be an empty line
    f.readline()

    # put points data into numpy's array
    data_sheet_list.append(np.zeros((count, dimension + 1)))
    for i in range(0, count):
        crt_grid_point = [float(x) for x in f.readline().split()]
        data_sheet_list[k][i, :] = np.array(crt_grid_point)


# plot
figure_path = os.path.join(file_dir, "figure/" + data_name_list[0] + "_cp")
fig, ax = plt.subplots()

for data_sheet, data_name in zip(data_sheet_list, data_name_list):
    ax.plot(data_sheet[:, 0], data_sheet[:, 1], linewidth=2.0, label=data_name)

plt.savefig(figure_path)

for f in file_obj_list:
    f.close()