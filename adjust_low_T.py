#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'


from Models import fastIsing
from Toolbox import simulation
from Utils import IO
import networkx as nx, itertools, scipy,\
        os, pickle, h5py, sys, multiprocessing as mp, json,\
        datetime, sys
import time
import timeit
from matplotlib.pyplot import *
from numpy import *
from tqdm import tqdm
from scipy import optimize, ndimage
import glob

def sig(x, b, d, c):
    return 1 / (1 + np.exp(b*(x-d))) + c

if __name__ == '__main__':

    for k in [1.96, 2.28, 2.80, 2.88, 2.16, 2.72, 3.16, 3.24, 3.52, 3.64]:
        Tc_results = IO.TcResult.loadFromPickle('DataTc_new/small_graphs/N=50', f'ER_k={k:.2f}_N=50_Tc_results')

        temps = Tc_results.temps
        abs_mags = Tc_results.abs_mags
        mags = Tc_results.mags
        T_c = Tc_results.T_c

        a, b = optimize.curve_fit(sig, temps, abs_mags)
        mag_Tc = sig(T_c, *a)
        print(f'mag at Tc: {T_c} \t {mag_Tc} \t {mags[np.where(temps > T_c)[0][0]]}')
        highT = temps[np.where(sig(temps, *a) < 0.75 * mag_Tc)[0][0]]
        lowT = temps[np.where(sig(temps, *a) < (1 + 0.75) * mag_Tc)[0][0]]
        #print(f'magnetization minus 25%: {highT}\t {sig(highT, *a)} \t {mags[np.where(temps > highT)[0][0]]}')
        print(f'magnetization plus  30%: {lowT} \t {sig(lowT, *a)} \t {ndimage.filters.gaussian_filter1d(np.abs(mags), 5)[np.where(temps > lowT)[0][0]]}')

        #symmetry_breaking_idx = np.where(np.abs(ndimage.filters.gaussian_filter1d(np.abs(mags), 10) - ndimage.filters.gaussian_filter1d(np.abs(abs_mags), 10)) > 0.01)[0][0]
        #lowT = temps[np.where(sig(temps, *a) < (1 + sig(temps[symmetry_breaking_idx], *a)) * 0.5)[0][0]]

        lowT = temps[np.where(sig(temps, *a) < (1 + mag_Tc) * 0.5)[0][0]]

        Tc_results.T_low = lowT
        Tc_results.saveToPickle('DataTc_new/small_graphs/N=50')

        Tc_results = IO.TcResult.loadFromPickle('DataTc_new/small_graphs/N=50', f'ER_k={k:.2f}_N=50_Tc_results')
        print(f'new T_low = {Tc_results.T_low}')

        with open(os.path.join('DataTc_new/small_graphs/N=50', f'ER_k={k:.2f}_N=50_Tc.txt'), 'w') as f:
            f.write(f'{Tc_results.T_low:.2f}\n{Tc_results.T_c:.2f}\n{Tc_results.T_high:.2f}\n')
