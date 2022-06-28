#!/usr/bin/env python3

## 5*6 grid
import sys
import numpy as np

from MD_functions import EnergyMinimization
from DynamicalMatrix import DM
from PlotFunctions import ConfigPlot_DiffSize, EigenModePlot_DiffSize


inflate_id = int(sys.argv[1])
inflate_d = float(sys.argv[2])
y_top_disp = float(sys.argv[3])

n_col = 5
n_row = 5
N = (2 * n_col - 1) * int(np.floor(n_row / 2)) + n_col % 2 * n_col # total number of particles

d0 = 1.
Lx = d0 * n_col
Ly = (n_row - 1) * np.sqrt(3) / 2 * d0 + d0
Ly -= y_top_disp * d0

D = np.ones(N) * d0
D[inflate_id] = d0 * (1. + inflate_d) # inflate this specific particle
x0 = np.zeros(N)
y0 = np.zeros(N)

ind = -1
for i_row in range(n_row):
    if i_row % 2 == 0:
        n_col_now = n_col
    else:
        n_col_now = n_col - 1
    for i_col in range(n_col_now):
        ind += 1
        if i_row % 2 == 0:
            x0[ind] = (i_col + 0.5) * d0
        else:
            x0[ind] = (i_col + 1.) * d0
        y0[ind] = i_row * 0.5 * np.sqrt(3) * d0
y0 = y0 + 0.5 * d0

mass = np.ones(N)
k_list = np.ones(N) * 2.

x, y = EnergyMinimization(N, x0, y0, D, Lx, Ly, k_list)

ConfigPlot_DiffSize(N, x, y, D, Lx, Ly, k_list, 1, 1, f"./Config_{inflate_id+1:02d}.png")

w, v = DM(N, x, y, D, mass, Lx, Ly, k_list)

eig_idx = 0
print(w[eig_idx])

EigenModePlot_DiffSize(N, x, y, D, Lx, Ly, k_list, v[:, eig_idx], 1, 1, f"./EigenMode_{inflate_id+1:02d}_{eig_idx+1:03d}.png")

fp = open('./Pos.txt', 'w')

for data in D:
    fp.write('%.32e\n' % data)

for data in x:
    fp.write('%.32e\n' % data)

for data in y:
    fp.write('%.32e\n' % data)

fp.write('%.32e\n' % Lx)
fp.write('%.32e\n' % Ly)
fp.close()