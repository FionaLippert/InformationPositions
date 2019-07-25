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
import argparse

parser = argparse.ArgumentParser(description='find the critical temperature of Ising Model on a given network structure')
parser.add_argument('dir', type=str, help='target directory')
parser.add_argument('graph', type=str, help='path to graph or ensemble of graphs')
parser.add_argument('--minT', type=float, default=0.1, help='minimum temperature')
parser.add_argument('--maxT', type=float, default=5.0, help='maximum temperature')
parser.add_argument('--numT', type=int, default=1000, help='number of different temperature values')
#parser.add_argument('--magFrac', type=float, default=0.3, help='fraction by which magnetization at T_c should be increased to reach the symmetriy breaking regime')


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
    sus_smoothed = ndimage.filters.gaussian_filter1d(sus, sigma)
    max_idx = np.argmax(sus_smoothed)
    Tc = temps[max_idx]
    T_susHalf = temps[max_idx:][np.where(sus_smoothed[max_idx:] < 0.5 * sus_smoothed[max_idx])[0][0]] # T where sus drops below half of its maximum
    return Tc, T_susHalf

def sig(x, b, d, c):
    return 1 / (1 + np.exp(b*(x-d))) + c


if __name__ == '__main__':

    args = parser.parse_args()

    ensemble = [g for g in glob.iglob(f'{args.graph}*.gpickle')]
    print(ensemble)

    #temps = linspace(1, 50, 500)
    #temps = linspace(1, 2, 100)
    #temps = linspace(0.1, 5, 1000)
    temps = linspace(args.minT, args.maxT, args.numT)
    nSamples      = int(1e4) #int(1e6)
    burninSamples = int(1e4) # int(1e6)
    magSide       = '' # which sign should the overall magnetization have (''--> doesn't matter, 'neg' --> flip states if <M> > 0, 'pos' --> flip if <M> < 0)
    updateType    = ''


    targetDirectory = f'{os.getcwd()}/{args.dir}'
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


        mags, sus, binder, abs_mags = simulation.magnetizationParallel(model,       \
                            temps           = temps,        \
                            n               = nSamples,     \
                            burninSamples   = burninSamples)


        T_c, T_susHalf = find_Tc_gaussian(sus, temps)
        print(f'Tc = {T_c}, T_susHalf = {T_susHalf}')

        #print(mags)
        #print(abs_mags)


        a, b = optimize.curve_fit(sig, temps, abs_mags)
        mag_Tc = sig(T_c, *a)
        print(f'mag at Tc: {T_c} \t {mag_Tc} \t {mags[np.where(temps > T_c)[0][0]]}')
        highT = temps[np.where(sig(temps, *a) < 0.75 * mag_Tc)[0][0]]

        """
        lowT = temps[np.where(sig(temps, *a) < (1 + args.magFrac) * mag_Tc)[0][0]]
        #print(f'magnetization minus 25%: {highT}\t {sig(highT, *a)} \t {mags[np.where(temps > highT)[0][0]]}')
        print(f'magnetization plus  30%: {lowT} \t {sig(lowT, *a)} \t {ndimage.filters.gaussian_filter1d(np.abs(mags), 5)[np.where(temps > lowT)[0][0]]}')

        lowT = temps[np.where(sig(temps, *a) < (1 + mag_Tc) * 0.5)[0][0]]
        print(f'magnetization half: {lowT} \t {sig(lowT, *a)} \t {ndimage.filters.gaussian_filter1d(np.abs(mags), 5)[np.where(temps > lowT)[0][0]]}')
        """

        symmetry_breaking_idx = np.where(np.abs(ndimage.filters.gaussian_filter1d(np.abs(mags), 10) - ndimage.filters.gaussian_filter1d(np.abs(abs_mags), 10)) > 0.01)[0][0]
        lowT = temps[np.where(sig(temps, *a) < (1 + sig(temps[symmetry_breaking_idx], *a)) * 0.5)[0][0]]

        highT = temps[np.where(sig(temps, *a) < sig(temps[symmetry_breaking_idx], *a) * 0.5)[0][0]]

        mag70 = temps[np.where(sig(temps, *a) < 0.7)[0][0]]
        print(f'T_low = {lowT}, T_high = {highT}')

        """
        tmp = dict( \
                temps = temps, \
                magnetization = mags, \
                absMagnetization = abs_mags, \
                susceptibility = sus, \
                binder = binder, \
                Tc = T_c, \
                highT = T_susHalf, \
                lowT = lowT)
        IO.savePickle(targetDirectory, f'{filename}_results', tmp)
        """

        result = IO.TcResult(temps, mags, abs_mags, sus, binder, T_c, highT, lowT, filename)
        result.saveToPickle(targetDirectory)

        results2 = IO.TcResult.loadFromPickle(targetDirectory, f'{filename}_Tc_results')
        print(results2.T_c)

        with open(os.path.join(targetDirectory, f'{filename}_Tc.txt'), 'w') as f:
            f.write(f'{lowT:.2f} \n {T_c:.2f} \n {highT:.2f} \n')



    print(targetDirectory)
