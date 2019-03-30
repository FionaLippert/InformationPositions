#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'


from Models import fastIsing
from Toolbox import infcy
from Utils import IO
import networkx as nx, itertools, scipy,\
        os, pickle, h5py, sys, multiprocessing as mp, json,\
        datetime, sys
import time
import timeit
from matplotlib.pyplot import *
from numpy import *
from tqdm import tqdm
from scipy import sparse
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


if __name__ == '__main__':

    network_path = sys.argv[1] # e.g. 'ER/ER_k=2.5_N=500'
    ensemble_size = len([g for g in glob.iglob(f'networkData/{network_path}*.gpickle')])
    #ensemble_size = 10
    all_Tc = np.zeros(ensemble_size)

    temps = linspace(0.5, 4, 10)
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
    for i in range(ensemble_size):
        file = f'networkData/{network_path}_v{i}.gpickle'
        graph = nx.read_gpickle(file)

        modelSettings = dict(\
                             graph       = graph,\
                             updateType  = updateType,\
                             magSide     = magSide
                             )
        model = fastIsing.Ising(**modelSettings)


        mag, sus, binder = infcy.magnetizationParallel(model,       \
                            temps           = temps,        \
                            n               = nSamples,     \
                            burninSamples   = burninSamples)

        #np.save(f'{targetDirectory}/mags_v{i}.npy', mag)
        #np.save(f'{targetDirectory}/susceptibility_v{i}.npy', sus)
        #np.save(f'{targetDirectory}/binder_v{i}.npy', binder)

        Tc = find_Tc(sus, temps)
        print(Tc)
        all_Tc[i] = Tc

        tmp = dict( \
                temps = temps, \
                magnetization = mag, \
                susceptibility = sus, \
                binder = binder, \
                Tc = Tc)
        IO.savePickle(f'{targetDirectory}/results_v{i}.pickle', tmp)

        #fig, ax = subplots(figsize=(10,6))
        #ax.scatter(temps, mag, alpha = .2, label='magnetization')
        #ax.scatter(temps, sus, alpha = .2, label='susceptibility')
        #ax.scatter(temps, binder, alpha = .2, label='Binder cumulant')
        #ax.axvline(x=Tc, ls='--', label=r'$T_c$')
        #ax.legend()
        #setp(ax, **dict(xlabel = 'Temperature', ylabel = '<M>'))
        #savefig(f'{targetDirectory}/temp_vs_mag.png')

    with open(f'{targetDirectory}/avg_Tc.txt', 'w') as f:
        f.write(f'{np.mean(all_Tc):.2f}')

    np.save(f'{targetDirectory}/all_Tc.npy', all_Tc)

    print(targetDirectory)
