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

data_name = sys.argv[1]
file_dir = os.path.dirname(os.path.realpath('__file__'))
data_path = os.path.join(file_dir, "output/" + data_name + ".wavedata")

with open(data_path, 'r', encoding="utf-8") as f:
    meta_str = f.readline()
    meta_data = [int(x) for x in meta_str.split()[1:]]

    dimension = meta_data[0]
    count = meta_data[1]
    grid_mode = meta_data[2]

    # the first line after metadata should be an empty line
    f.readline()

    # put points data into numpy's array
    data_sheet = np.zeros((count, dimension + 1))
    for i in range(0, count):
        crt_grid_point = [float(x) for x in f.readline().split()]
        data_sheet[i, :] = np.array(crt_grid_point)

    # plot
    figure_path = os.path.join(file_dir, "figure/" + data_name)

    if dimension == 1:
        fig, ax = plt.subplots()
        ax.plot(data_sheet[:, 0], data_sheet[:, 1], linewidth=2.0)
        plt.savefig(figure_path)

    elif dimension == 2:
        count1, count2 = meta_data[3], meta_data[4]
        
        rs = np.reshape(data_sheet[:, 0], (count2, count1))
        thetas = np.reshape(data_sheet[:, 1], (count2, count1))
        # results = np.log(np.reshape(data_sheet[:, 2], (count2, count1)) + 1e-8)
        results = np.reshape(data_sheet[:, 2], (count2, count1))

        # rs = rs[:, 0: r_max_count]
        # thetas = thetas[:, 0: r_max_count]
        # results = results[:, 0: r_max_count]

        level = np.linspace(results.min(), results.max(), 50)
        
        if grid_mode == POLAR:
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
            cs = ax.contourf(thetas, rs, results, levels=level)
        elif grid_mode == XYZ:
            fig, ax = plt.subplots()
            cs = ax.contourf(rs, thetas, results, levels=level)
        
        plt.savefig(figure_path)
            

    elif dimension == 3:
        #count1, count2, count3 = meta_data[3], meta_data[4], meta_data[5]
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter(xs=data_sheet[:, 0], ys=data_sheet[:, 1], zs=data_sheet[:, 2], c=data_sheet[:, 3], cmap='Greens', s=1, alpha=0.01)
        plt.savefig(figure_path)
