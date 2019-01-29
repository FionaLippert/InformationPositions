#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Casper van Elteren'
"""
Created on Mon Jun 11 09:06:57 2018

@author: casper
"""

from Models import fastIsing
from Toolbox import infcy
from Utils import IO, plotting as plotz
from Utils.IO import SimulationResult
import networkx as nx, itertools, scipy,\
        os, pickle, h5py, sys, multiprocessing as mp, json,\
        datetime, sys
import time
import timeit
from matplotlib.pyplot import *
from numpy import *
from tqdm import tqdm
from functools import partial
from scipy import sparse
close('all')


if __name__ == '__main__':

    # 2e4 steps with non-single updates and 32x32 grid --> serial-time = parallel-time

    nSamples      = int(2e4) #int(1e6)
    burninSamples = 0#int(1e3) # int(1e6)
    magSide       = '' # which sign should the overall magnetization have (''--> doesn't matter, 'neg' --> flip states if <M> > 0, 'pos' --> flip if <M> < 0)
    updateType    = ''
    CHECK         = .5  #[.8, .5, .2]   # value of 0.8 means match magnetiztion at 80 percent of max


    graph = nx.grid_2d_graph(16, 16, periodic=True)

    now = time.time()
    targetDirectory = f'{os.getcwd()}/Data/{now}'
    os.mkdir(targetDirectory)

    settings = dict(
        nSamples         = nSamples,\
        burninSamples    = burninSamples,\
        updateMethod     = updateType,\
        nNodes           = graph.number_of_nodes()
        )
    IO.saveSettings(targetDirectory, settings)

    # graph = nx.barabasi_albert_graph(10, 3)
    modelSettings = dict(\
                         graph       = graph,\
                         temperature = 0,\
                         updateType  = updateType,\
                         magSide     = magSide
                         )
    model = fastIsing.Ising(**modelSettings)
    updateType = model.updateType

    # match the temperature to sample from
    if os.path.isfile(f'{targetDirectory}/mags.pickle'):
        tmp = IO.loadPickle(f'{targetDirectory}/mags.pickle')
        for i, j in tmp.items():
            globals()[i] = j
    else:
        magRange = array([CHECK]) if isinstance(CHECK, float) else array(CHECK) # ratio of magnetization to be reached
        temps = linspace(0.1, 5, 50)
        print(temps)

        """
        start = time.process_time()
        mag, sus = infcy.magnetizationParallel(model,\
                        temps = temps,\
                        n = nSamples, burninSamples = burninSamples)
        end = time.process_time()
        print("parallel: {}".format(end-start))

        print(temps)
        """
        """
        start = time.process_time()
        mag, sus = model.matchMagnetization(\
                        temps = temps,\
                        n = nSamples, burninSamples = burninSamples)
        end = time.process_time()
        print("serial: {}".format(end-start))
        """
        """
        func = lambda x, a, b, c, d :  a / (1 + exp(b * (x - c))) + d # tanh(-a * x)* b + c
        # func = lambda x, a, b, c : a + b*exp(-c * x)
        print(temps, mag.squeeze())
        a, b = scipy.optimize.curve_fit(func, temps, mag.squeeze(), maxfev = 10000)

        # run the simulation per temperature
        temperatures = array([])
        f_root = lambda x,  c: func(x, *a) - c
        magnetizations = max(mag) * magRange
        for m in magnetizations:
            r = scipy.optimize.root(f_root, 0, args = (m), method = 'linearmixing')
            rot = r.x if r.x > 0 else 0
            temperatures = hstack((temperatures, rot))

        fig, ax = subplots()
        xx = linspace(0, max(temps), 1000)
        ax.plot(xx, func(xx, *a))
        ax.scatter(temperatures, func(temperatures, *a), c ='red')
        ax.scatter(temps, mag, alpha = .2)
        setp(ax, **dict(xlabel = 'Temperature', ylabel = '<M>'))
        savefig(f'{targetDirectory}/temp_vs_mag.png')

        tmp = dict(temps = temps, \
        temperatures = temperatures, magRange = magRange, mag = mag)
        IO.savePickle(f'{targetDirectory}/mags.pickle', tmp)
        """

        infcy.collectSnapshots(model, repeats=100, burninSamples=int(1e3), nSamples=int(10), distSamples=int(1e3))
