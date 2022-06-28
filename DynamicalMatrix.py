#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def DM(N, x0, y0, D0, m0, Lx, Ly, k_list):
    
    M = np.zeros((2*N, 2*N))

    for i in range(N):
        r_now = 0.5 * D0[i]

        if x0[i] < r_now or Lx - x0[i] < r_now:
            M[2*i, 2*i] = M[2*i, 2*i] + k_list[i]
        if y0[i] < r_now or Ly - y0[i] < r_now:
            M[2*i+1, 2*i+1] = M[2*i+1, 2*i+1] + k_list[i]

        for j in range(i):
            dij = 0.5 * (D0[i] + D0[j])
            dijsq = dij**2
            dx = x0[i] - x0[j]
            dy = y0[i] - y0[j]
            rijsq = dx**2 + dy**2
            if rijsq < dijsq:
                k = k_list[i] * k_list[j] / (k_list[i] + k_list[j])
                rijmat = np.array([[dx * dx, dx * dy], [dx * dy, dy * dy]])
                rij = np.sqrt(rijsq)
                Mij1 = -k * rijmat / rijsq
                Mij2 = -k * (dij - rij) * (rijmat / rijsq - [[1, 0], [0, 1]]) / rij
                Mij = Mij1 + Mij2
                M[2*i:2*i+2,2*j:2*j+2] = Mij
                M[2*j:2*j+2,2*i:2*i+2] = Mij
                M[2*i:2*i+2,2*i:2*i+2] = M[2*i:2*i+2,2*i:2*i+2] - Mij
                M[2*j:2*j+2,2*j:2*j+2] = M[2*j:2*j+2,2*j:2*j+2] - Mij

    m_sqrt = np.zeros((2*N, 2*N))
    for i in range(N):
        m_sqrt[2*i, 2*i] = 1 / np.sqrt(m0[i])
        m_sqrt[2*i+1, 2*i+1] = 1 / np.sqrt(m0[i])

    M = m_sqrt.dot(M).dot(m_sqrt)

    w, v = np.linalg.eig(M)

    # sort eigenvalues and associated eigenvectors from small to big
    idx = w.argsort()
    w = w[idx]
    v = v[:, idx]
    
    return w,v