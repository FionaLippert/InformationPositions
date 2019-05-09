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


if __name__ == '__main__':

    network_path = sys.argv[1] # e.g. 'ER/ER_k=2.5_N=500'
    print(network_path)

    ensemble = [g for g in glob.iglob(f'networkData/{network_path}*.gpickle')]
    print(ensemble)
    #ensemble_size = 10
    all_Tc = np.zeros(len(ensemble))

    #temps = linspace(1, 50, 500)
    #temps = linspace(0.1, 4.0, 500)
    temps = linspace(0.5, 5, 500)
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


        mag, sus, binder = infcy.magnetizationParallel(model,       \
                            temps           = temps,        \
                            n               = nSamples,     \
                            burninSamples   = burninSamples)

        #np.save(f'{targetDirectory}/mags_v{i}.npy', mag)
        #np.save(f'{targetDirectory}/susceptibility_v{i}.npy', sus)
        #np.save(f'{targetDirectory}/binder_v{i}.npy', binder)

        #func = lambda x, a, b, c:  a / (1 + np.exp(b * (x - c)))
        #try:
        #binder[np.where(binder < 0)] = np.nan
        #valid = np.where(np.isfinite(binder))
        #np.save(f'{targetDirectory}/sus.npy', sus)
        #params, _ = optimize.curve_fit(func, temps[valid], binder[valid], p0=[0.7, 5., 2.])
        #idx = np.where(func(temps, *params) < params[0] - params[0]/10.)[0][0]
        #T_c_estimate = temps[idx]
        #idx = np.where(np.abs(temps - T_c_estimate) < 0.5)
        #print(idx)

        #Tc = find_Tc(sus[idx], temps[idx])
        #except:
        #    Tc = np.nan
        #print(Tc)
        Tc = find_Tc_gaussian(sus, temps)
        print(f'Tc = {Tc}')
        all_Tc[i] = Tc
        #print(sus)

        tmp = dict( \
                temps = temps, \
                magnetization = mag, \
                susceptibility = sus, \
                binder = binder, \
                Tc = Tc)
        IO.savePickle(targetDirectory, f'{filename}_results', tmp)

        with open(os.path.join(targetDirectory, f'{filename}_Tc.txt'), 'w') as f:
            f.write(f'{Tc:.2f}')


        """
        temps_Tc = linspace(Tc - 0.25, Tc + 0.25, 500)
        mag_Tc, sus_Tc, binder_Tc = infcy.magnetizationParallel(model,       \
                            temps           = temps_Tc,        \
                            n               = nSamples,     \
                            burninSamples   = burninSamples)

        Tc_refined = find_Tc_gaussian(sus_Tc, temps_Tc)
        print(Tc_refined)

        tmp = dict( \
                temps = temps, \
                magnetization = mag, \
                susceptibility = sus, \
                binder = binder, \
                Tc = Tc, \
                temps_refined = temps_Tc, \
                magnetization_refined = mag_Tc, \
                susceptibility_refined = sus_Tc, \
                binder_refined = binder_Tc, \
                Tc_refined = Tc_refined)
        IO.savePickle(os.path.join(targetDirectory, f'{filename}_results.pickle'), tmp)

        with open(os.path.join(targetDirectory, f'{filename}_Tc_refined.txt'), 'w') as f:
            f.write(f'{Tc_refined:.2f}')
        """

        #fig, ax = subplots(figsize=(10,6))
        #ax.scatter(temps, mag, alpha = .2, label='magnetization')
        #ax.scatter(temps, sus, alpha = .2, label='susceptibility')
        #ax.scatter(temps, binder, alpha = .2, label='Binder cumulant')
        #ax.axvline(x=Tc, ls='--', label=r'$T_c$')
        #ax.legend()
        #setp(ax, **dict(xlabel = 'Temperature', ylabel = '<M>'))
        #savefig(f'{targetDirectory}/temp_vs_mag.png')

    with open(f'{targetDirectory}/avg_Tc.txt', 'w') as f:
        f.write(f'{np.nanmean(all_Tc):.2f}')

    np.save(f'{targetDirectory}/all_Tc.npy', all_Tc)

    print(targetDirectory)
