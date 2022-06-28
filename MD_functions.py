#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import time
#from numba import jit

def Force(N, x, y, D, Lx, Ly, k_list):
    Fx = np.zeros(N)
    Fy = np.zeros(N)
    F_bottom = []
    for n in range(N):
        r_now = 0.5 * D[n]
        
        if x[n] < r_now:
            Fx[n] += k_list[n] * (r_now - x[n])
        elif x[n] > Lx - r_now:
            Fx[n] -= k_list[n] * (r_now - Lx + x[n])
        
        if y[n] < r_now:
            Fy[n] += k_list[n] * (r_now - y[n])
            F_bottom.append(-k_list[n] * (r_now - y[n]))
        elif y[n] > Ly - r_now:
            Fy[n] -= k_list[n] * (r_now - Ly + y[n])

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
                        F = -k * (Dmn - dmn) / dmn
                        dFx = F * dx
                        dFy = F * dy
                        Fx[n] += dFx
                        Fx[m] -= dFx
                        Fy[n] += dFy
                        Fy[m] -= dFy

    return Fx, Fy, F_bottom


def Force_VL(N, x, y, D, Lx, Ly, k_list, VL_list, VL_counter):
    Fx = np.zeros(N)
    Fy = np.zeros(N)
    Ep = 0
    for n in range(N):
        r_now = 0.5 * D[n]
        
        if x[n] < r_now:
            dmn = x[n]
            Fx[n] += k_list[n] * (r_now - dmn)
            Ep += 0.5 * k_list[n] * (r_now - dmn)**2
        elif x[n] > Lx - r_now:
            dmn = Lx - x[n]
            Fx[n] -= k_list[n] * (r_now - dmn)
            Ep += 0.5 * k_list[n] * (r_now - dmn)**2
        
        if y[n] < r_now:
            dmn = y[n]
            Fy[n] += k_list[n] * (r_now - dmn)
            Ep += 0.5 * k_list[n] * (r_now - dmn)**2
        elif y[n] > Ly - r_now:
            dmn = Ly - y[n]
            Fy[n] -= k_list[n] * (r_now - dmn)
            Ep += 0.5 * k_list[n] * (r_now - dmn)**2

    for vl_idx in np.arange(VL_counter):
        n = VL_list[vl_idx][0]
        m = VL_list[vl_idx][1]
        dy = y[m] - y[n]
        Dmn = 0.5 * (D[m] + D[n])
        if abs(dy) < Dmn:
            dx = x[m] - x[n]
            if abs(dx) < Dmn:
                dmn = np.sqrt(dx**2 + dy**2)
                if dmn < Dmn:
                    k = k_list[n] * k_list[m] / (k_list[n] + k_list[m])
                    F = -k * (Dmn - dmn) / dmn
                    dFx = F * dx
                    dFy = F * dy
                    Fx[n] += dFx
                    Fx[m] -= dFx
                    Fy[n] += dFy
                    Fy[m] -= dFy
                    Ep += 0.5 * k * (Dmn - dmn)**2

    return Fx, Fy, Ep


def VerletList(N, x, y, D, VL_list, VL_counter_old, x_save, y_save, first_call):    
    
    r_factor = 1.2
    r_cut = np.amax(D)
    r_list = r_factor * r_cut
    r_list_sq = r_list**2
    r_skin_sq = ((r_factor - 1.0) * r_cut)**2

    if first_call == 0:
        dr_sq_max = 0.0
        for n in np.arange(N):
            dy = y[n] - y_save[n]
            dx = x[n] - x_save[n]
            dr_sq = dx**2 + dy**2
            if dr_sq > dr_sq_max:
                dr_sq_max = dr_sq
        if dr_sq_max < r_skin_sq:
            return VL_list, VL_counter_old, x_save, y_save

    VL_counter = 0
    
    for n in np.arange(N):
        for m in np.arange(n+1, N):
            dy = y[m] - y[n]
            Dmn = 0.5 * (D[m] + D[n])
            if abs(dy) < r_list:
                dx = x[m] - x[n]
                if abs(dx) < r_list:
                    dmn_sq = dx**2 + dy**2
                    if dmn_sq < r_list_sq:
                        VL_list[VL_counter][0] = n
                        VL_list[VL_counter][1] = m
                        VL_counter += 1

    return VL_list, VL_counter, x, y


def MD_SD(N, x0, y0, D0, Lx, Ly, k_list):    
    t_start = time.time()

    Fthresh = 1e-11
    dt = np.sqrt(k_list[2]) / 20.0
    Nt = 1000000
    
    vx = np.zeros(N)
    vy = np.zeros(N)    
    x = np.array(x0)
    y = np.array(y0)
    x_save = np.array(x0)
    y_save = np.array(y0)

    VL_list = np.zeros((N * 10, 2), dtype = int) 
    VL_counter = 0
    VL_list, VL_counter, x_save, y_save = VerletList(N, x, y, D0, Lx, VL_list, VL_counter, x_save, y_save, 1)
    
    for nt in np.arange(Nt):
        VL_list, VL_counter, x_save, y_save = VerletList(N, x, y, D0, Lx, VL_list, VL_counter, x_save, y_save, 0)
        Fx, Fy, Ep_now = Force_VL(N, x, y, D0, Lx, Ly, k_list, VL_list, VL_counter)
        vx = dt * Fx
        vy = dt * Fy
        x += vx * dt
        y += vy * dt
        F_tot = np.mean(np.sqrt(Fx**2 + Fy**2))
        # putting a threshold on total force
        if F_tot < Fthresh:
            break
    
    t_end = time.time()
    print("Total Time Step: %d" % nt)
    print("Mean Force: %.3e" % F_tot)
    print("time = %.3e" %(t_end-t_start))

    return x, y, Ep_now


def FIRE(N, x0, y0, D0, Lx, Ly, k_list):  
    t_start = time.time()
    # FIRE parameters
    Fthresh = 1e-14
    dt_md = 0.01 * np.sqrt(max(k_list))
    Nt = 1000000 # maximum fire md steps
    N_delay = 20
    N_pn_max = 2000
    f_inc = 1.1
    f_dec = 0.5
    a_start = 0.15
    f_a = 0.99
    dt_max = 10.0 * dt_md
    dt_min = 0.05 * dt_md
    initialdelay = 1
    
    vx = np.zeros(N)
    vy = np.zeros(N)    
    x = np.array(x0)
    y = np.array(y0)
    x_save = np.array(x0)
    y_save = np.array(y0)

    VL_list = np.zeros((N * 10, 2), dtype = int) 
    VL_counter = 0
    VL_list, VL_counter, x_save, y_save = VerletList(N, x, y, D0, VL_list, VL_counter, x_save, y_save, 1)
    Fx, Fy, Ep_now = Force_VL(N, x, y, D0, Lx, Ly, k_list, VL_list, VL_counter)
    F_tot = np.mean(np.sqrt(Fx**2 + Fy**2))
    # putting a threshold on total force
    if F_tot < Fthresh:
        return x, y, Ep_now
        
    a_fire = a_start
    delta_a_fire = 1.0 - a_fire
    dt = dt_md
    dt_half = dt / 2.0

    N_pp = 0 # number of P being positive
    N_pn = 0 # number of P being negative
    ## FIRE
    for nt in np.arange(Nt):
        # FIRE update
        P = np.dot(vx, Fx) + np.dot(vy, Fy)
        
        if P > 0.0:
            N_pp += 1
            N_pn = 0
            if N_pp > N_delay:
                dt = min(f_inc * dt, dt_max)
                dt_half = dt / 2.0
                a_fire = f_a * a_fire
                delta_a_fire = 1.0 - a_fire
        else:
            N_pp = 0
            N_pn += 1
            if N_pn > N_pn_max:
                break
            if (initialdelay < 0.5) or (nt >= N_delay):
                if f_dec * dt > dt_min:
                    dt = f_dec * dt
                    dt_half = dt / 2.0
                a_fire = a_start
                delta_a_fire = 1.0 - a_fire
                x -= vx * dt_half
                y -= vy * dt_half
                vx = np.zeros(N)
                vy = np.zeros(N)

        # MD using Verlet method
        vx += Fx * dt_half
        vy += Fy * dt_half
        rsc_fire = np.sqrt(np.sum(vx**2 + vy**2)) / np.sqrt(np.sum(Fx**2 + Fy**2))
        vx = delta_a_fire * vx + a_fire * rsc_fire * Fx
        vy = delta_a_fire * vy + a_fire * rsc_fire * Fy
        x += vx * dt
        y += vy * dt

        VL_list, VL_counter, x_save, y_save = VerletList(N, x, y, D0, VL_list, VL_counter, x_save, y_save, 0)
        Fx, Fy, Ep_now = Force_VL(N, x, y, D0, Lx, Ly, k_list, VL_list, VL_counter)

        F_tot = np.mean(np.sqrt(Fx**2 + Fy**2))
        # putting a threshold on total force
        if F_tot < Fthresh:
            break

        vx += Fx * dt_half
        vy += Fy * dt_half

    t_end = time.time()
    print("Total Time Step: %d" % nt)
    print("Mean Force: %.3e" % F_tot)
    print("time = %.3e" %(t_end-t_start))

    return x, y, Ep_now


def EnergyMinimization(N, x0, y0, D, Lx, Ly, k_list, method = 'FIRE'):

    if method == 'FIRE':
        x, y, Ep = FIRE(N, x0, y0, D, Lx, Ly, k_list)
    else:
        x, y, Ep = MD_SD(N, x0, y0, D, Lx, Ly, k_list)

    return x, y