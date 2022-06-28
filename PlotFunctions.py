#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def Force(N, x, y, D, Lx, Ly, k_list):
    xc = []
    yc = []
    Fc = []

    for n in np.arange(N):
        r_now = 0.5 * D[n]
        
        if x[n] < r_now:
            xc.append([0., x[n]])
            yc.append([y[n], y[n]])
            Fc.append(k_list[n] * (1 - x[n] / r_now) / r_now)
        elif x[n] > Lx - r_now:
            xc.append([x[n], Lx])
            yc.append([y[n], y[n]])
            Fc.append(k_list[n] * (1 - (Lx - x[n]) / r_now) / r_now)
        
        if y[n] < r_now:
            xc.append([x[n], x[n]])
            yc.append([0., y[n]])
            Fc.append(k_list[n] * (1 - y[n] / r_now) / r_now)
        elif y[n] > Ly - r_now:
            xc.append([x[n], x[n]])
            yc.append([y[n], Ly])
            Fc.append(k_list[n] * (1 - (Ly - y[n]) / r_now) / r_now)

    for n in range(N - 1):
        for m in range(n+1, N):
            dy = y[m] - y[n]
            Dmn = 0.5 * (D[m] + D[n])
            if abs(dy) < Dmn:
                dx = x[m] - x[n]
                if abs(dx) < Dmn:
                    dmn = np.sqrt(dx**2 + dy**2)
                    if dmn < Dmn:
                        k = k_list[n] * k_list[m] / (k_list[n] + k_list[m])
                        xc.append([x[m], x[n]])
                        yc.append([y[m], y[n]])
                        Fc.append(k * (1. - dmn / Dmn) / Dmn / dmn)

    return xc, yc, Fc

def ConfigPlot(N, x, y, D, Lx, Ly, k_list, cn_on = 1, mark_print = 0, fn = ''):

    fig, ax = plt.subplots(subplot_kw = {'aspect': 'equal'})

    # plot disks     
    for i in range(N):
        ax.add_patch(plt.Circle((x[i], y[i]), 0.5 * D[i], \
            facecolor = 'green', edgecolor = 'none', alpha = 0.5))

    # plot contact network, linewidth proportional to force magnitude
    if cn_on == 1:
        xc, yc, Fc = Force(N, x, y, D, Lx, Ly, k_list)
        Fmin = min(Fc)
        F_span = max(Fc) - Fmin
        for i in range(len(xc)):
            ax.plot(xc[i], yc[i], color = 'k',  linewidth = 1.5 + (Fc[i] - Fmin) / F_span * 1.0)
                
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
        label.set_font('helvetica')
    plt.xlabel('x', font = 'helvetica', fontsize = 16)
    plt.ylabel('y', font = 'helvetica', fontsize = 16)

    if (mark_print == 1) and (not fn):
        fig.savefig(fn, dpi = 300)

    plt.show()

def ConfigPlot_DiffSize(N, x, y, D, Lx, Ly, k_list, cn_on = 1, mark_print = 0, fn = ''):

    Dmin = np.mean(D)
    fig, ax = plt.subplots(subplot_kw = {'aspect': 'equal'})

    # plot disks         
    for i in range(N):
        if D[i] > Dmin:
            ax.add_patch(plt.Circle((x[i], y[i]), 0.5 * D[i], \
                facecolor = 'green', edgecolor = 'none', alpha = 0.7))
        else:
            ax.add_patch(plt.Circle((x[i], y[i]), 0.5 * D[i], \
                facecolor = 'green', edgecolor = 'none', alpha = 0.3))

    # plot contact network, linewidth proportional to force magnitude
    if cn_on == 1:
        xc, yc, Fc = Force(N, x, y, D, Lx, Ly, k_list)
        Fmin = min(Fc)
        F_span = max(Fc) - Fmin
        for i in range(len(xc)):
            ax.plot(xc[i], yc[i], color = 'k',  linewidth = 1.5 + (Fc[i] - Fmin) / F_span * 1.0)
                
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
        label.set_font('helvetica')
    plt.xlabel('x', font = 'helvetica', fontsize = 16)
    plt.ylabel('y', font = 'helvetica', fontsize = 16)

    # save figure
    if (mark_print == 1) and (not fn == ''):
        fig.savefig(fn, dpi = 300)

    plt.show()

def EigenModePlot(N, x, y, D, Lx, Ly, k_list, eigVec, cn_on = 1, mark_print = 0, fn = ''):
    
    fig, ax = plt.subplots(subplot_kw = {'aspect': 'equal'})
               
    # plot disks
    for i in range(N):
        ax.add_patch(plt.Circle((x[i], y[i]), 0.5 * D[i], \
            facecolor = 'green', edgecolor = 'none', alpha = 0.5))

    # plot eigenvectors
    vx = eigVec[0:2*N:2]
    vy = eigVec[1:2*N:2]
    ax.quiver(x, y, vx, vy, color = 'r')

    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
        label.set_font('helvetica')
    plt.xlabel('x', font = 'helvetica', fontsize = 16)
    plt.ylabel('y', font = 'helvetica', fontsize = 16)

    # save figure
    if (mark_print == 1) and (not fn == ''):
        fig.savefig(fn, dpi = 300)

    plt.show()


def EigenModePlot_DiffSize(N, x, y, D, Lx, Ly, k_list, eigVec, cn_on = 1, mark_print = 0, fn = ''):
    
    Dmin = np.mean(D)
    fig, ax = plt.subplots(subplot_kw = {'aspect': 'equal'})

    # plot disks         
    for i in range(N):
        if D[i] > Dmin:
            ax.add_patch(plt.Circle((x[i], y[i]), 0.5 * D[i], \
                facecolor = 'green', edgecolor = 'none', alpha = 0.7))
        else:
            ax.add_patch(plt.Circle((x[i], y[i]), 0.5 * D[i], \
                facecolor = 'green', edgecolor = 'none', alpha = 0.3))
              
    # plot eigenvectors
    vx = eigVec[0:2*N:2]
    vy = eigVec[1:2*N:2]
    ax.quiver(x, y, vx, vy, color = 'r')

    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
        label.set_font('helvetica')
    plt.xlabel('x', font = 'helvetica', fontsize = 16)
    plt.ylabel('y', font = 'helvetica', fontsize = 16)

    # save figure
    if (mark_print == 1) and (not fn == ''):
        fig.savefig(fn, dpi = 300)

    plt.show()