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
close('all')

# remove values that deviate more than one std from the median
def remove_outliers(data, f=1):
    m = np.median(data)
    std = np.std(data)
    d = np.abs(data - m)
    data[np.where(d >f*std)] = np.nan
    return data

def find_Tc(sus, temps, n=10, f=1):
    idx = np.argsort(sus)[-n:]
    return np.nanmean(remove_outliers(temps[idx], f))

def find_Tc_gaussian(sus, temps, sigma=5):
    return temps[np.argmax(ndimage.filters.gaussian_filter1d(sus, sigma))]

def sig(x, b, d, c):
    return 1 / (1 + np.exp(b*(x-d))) + c


if __name__ == '__main__':

    network_path = sys.argv[1] # e.g. 'ER/ER_k=2.5_N=500'
    print(network_path)

    ensemble = [g for g in glob.iglob(f'networkData/{network_path}*.gpickle')]
    print(ensemble)

    #temps = linspace(1, 50, 500)
    #temps = linspace(1, 2, 100)
    temps = linspace(0.1, 5, 1000)
    nSamples      = int(1e4) #int(1e6)
    burninSamples = int(1e4) # int(1e6)
    magSide       = '' # which sign should the overall magnetization have (''--> doesn't matter, 'neg' --> flip states if <M> > 0, 'pos' --> flip if <M> < 0)
    updateType    = ''


    targetDirectory = f'{os.getcwd()}/DataTc/{network_path}'
    os.makedirs(targetDirectory, exist_ok=True)

    settings = dict(
        nSamples         = nSamples, \
        burninSamples    = burninSamples, \
        updateMethod     = updateType
        )
    IO.saveSettings(targetDirectory, settings)

    #for file in glob.iglob(f'networkData/{network_path}*.gpickle'):
    for i, g in enumerate(ensemble):
        #file = f'networkData/{network_path}_v{i}.gpickle'
        graph = nx.read_gpickle(g)
        filename = os.path.split(g)[-1].strip('.gpickle')

        modelSettings = dict(\
                             graph       = graph,\
                             updateType  = updateType,\
                             magSide     = magSide
                             )
        model = fastIsing.Ising(**modelSettings)


        mag, sus, binder, mag_abs = simulation.magnetizationParallel(model,       \
                            temps           = temps,        \
                            n               = nSamples,     \
                            burninSamples   = burninSamples)


        Tc = find_Tc_gaussian(sus, temps)
        print(f'Tc = {Tc}')


        a, b = optimize.curve_fit(sig, temps, mag_abs)
        mag_Tc = sig(Tc, *a)
        print(f'mag at Tc = {mag_Tc}')
        highT = temps[np.where(sig(temps, *a) < 0.75 * mag_Tc)[0][0]]
        lowT = temps[np.where(sig(temps, *a) < 1.25 * mag_Tc)[0][0]]
        print(f'magnetization minus 25%: {highT} {sig(highT, *a)}')
        print(f'magnetization plus  25%: {lowT} {sig(lowT, *a)}')

        tmp = dict( \
                temps = temps, \
                magnetization = mag, \
                absMagnetization = mag_abs, \
                susceptibility = sus, \
                binder = binder, \
                Tc = Tc, \
                highT = highT, \
                lowT = lowT)
        IO.savePickle(targetDirectory, f'{filename}_results', tmp)

        with open(os.path.join(targetDirectory, f'{filename}_Tc.txt'), 'w') as f:
            f.write(f'{lowT:.2f} \n {Tc:.2f} \n {highT:.2f}')



    print(targetDirectory)
