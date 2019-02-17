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

    T             = 2.0
    nSamples      = int(1e4) #int(1e6)
    burninSamples = int(1e4) # int(1e6)
    magSide       = '' # which sign should the overall magnetization have (''--> doesn't matter, 'neg' --> flip states if <M> > 0, 'pos' --> flip if <M> < 0)
    updateType    = ''
    CHECK         = []  #[.8, .5, .2]   # value of 0.8 means match magnetiztion at 80 percent of max


    #graph = nx.grid_2d_graph(16, 16, periodic=True)
    avg_deg = 2.
    N = 500
    p = avg_deg/N

    """
    graph = nx.erdos_renyi_graph(N, p)
    connected_nodes = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(connected_nodes)
    nx.write_gpickle(graph, f'ER_avgDeg={avg_deg}_N={N}.gpickle', 2)
    print("avg degree = {}".format(np.mean([d for k, d in graph.degree()])))
    """
    """
    seq = nx.utils.powerlaw_sequence(N, 1.6)
    G = nx.expected_degree_graph(seq, selfloops = False)
    graph = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]
    nx.write_gpickle(graph, f'scaleFree_N={N}.gpickle')
    """

    graph = nx.read_gpickle("ER_avgDeg2.5_5000.gpickle")
    #graph = nx.read_gpickle("../WS_4_p02.gpickle")

    #diameter = nx.diameter(graph)
    #diameter = 14
    #print("diameter = {}".format(diameter))

    #graph = nx.read_gpickle('unweighted_person-person_projection_anonymous_combined_GC_stringToInt.gpickle')
    #connected_nodes = max(nx.connected_components(graph), key=len)
    #graph = graph.subgraph(connected_nodes)

    print(len(graph))

    now = time.time()
    targetDirectory = f'{os.getcwd()}/Data/{now}'
    os.mkdir(targetDirectory)

    settings = dict(
        T                = T,\
        nSamples         = nSamples,\
        burninSamples    = burninSamples,\
        updateMethod     = updateType,\
        nNodes           = graph.number_of_nodes()
        )
    IO.saveSettings(targetDirectory, settings)

    # graph = nx.barabasi_albert_graph(10, 3)
    modelSettings = dict(\
                         graph       = graph,\
                         temperature = T,\
                         updateType  = updateType,\
                         magSide     = magSide
                         )
    model = fastIsing.Ising(**modelSettings)
    updateType = model.updateType

    # match the temperature to sample from
    if len(sys.argv) > 1:
        if sys.argv[1] == "series":
            fig, ax = subplots(figsize=(10,5))
            for i in range(5):
                model.states = np.zeros(graph.number_of_nodes(), int, 'C') + 1 # start in 'all ones' state
                mags = infcy.magTimeSeries(model, burninSamples=int(0), nSamples=int(1e4))
                mags_normal = np.where(np.abs(mags)>0.05, mags, np.nan)
                mags_switch = np.where(np.abs(mags)<=0.05, mags, np.nan)
                ax.plot(mags)
                #ax.plot(mags_switch, c='r')
            savefig(f'{targetDirectory}/magsTimeSeries.png')
            np.save(f'{targetDirectory}/magsTimeSeries.npy', mags)
        if sys.argv[1] == "mixing":
            #model.states = np.zeros(graph.number_of_nodes(), int, 'C') + 1 # start in 'all ones' state
            mixingTimes = []
            corrTimes = []
            fig, ax = subplots(figsize=(10,5))
            fig2, ax2 = subplots(figsize=(10,5))
            for prob in tqdm(np.linspace(0.5, 1, 10)):
                #print(prob)
                model.states = np.random.choice([-1,1], size = model.nNodes, p=[prob, 1-prob])
                y, mixingTime, autocorr = infcy.determineMixingTime(model,\
                                      stepSizeBurnin = 10,\
                                      nStepsRegress = int(1e2),\
                                      threshold = 0.05,\
                                      nStepsCorr = int(1e4))

                #func = lambda x, t: np.exp(-x/t)
                #func = lambda x, m, c, c0: c0 + x**m * c
                #a, b = scipy.optimize.curve_fit(func, np.arange(autocorr.size), autocorr, p0=[-1., 1., 0.]) # characteristic autocorrelation time
                corrTime = np.where(np.abs(autocorr) < 0.05)[0][0]
                #mixingTimes.append(mixingTime)
                corrTimes.append(corrTime)
                ax.plot(autocorr[:corrTime*2])
                #xx = np.linspace(0, 500, 1000)
                ax.plot(corrTime, autocorr[corrTime], 'p', c='red')
                ax2.plot(y)

            #print(mixingTimes)
            print(corrTimes)

            #ax.plot(y)

            fig.savefig(f'{targetDirectory}/autocorr.png')
            fig2.savefig(f'{targetDirectory}/mag.png')
            #np.save(f'{targetDirectory}/mixing.npy', autocorr)
        print(targetDirectory)

    else:

        magRange = array([CHECK]) if isinstance(CHECK, float) else array(CHECK) # ratio of magnetization to be reached
        temps = linspace(0.1, 4, 16)
        #temps = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 5, 10])
        #print(temps)

        start = time.process_time()
        mag, sus = infcy.magnetizationParallel(model,\
                        temps = temps,\
                        n = nSamples, burninSamples = burninSamples)
        end = time.process_time()
        #print("parallel: {}".format(end-start))

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
        """
        fig, ax = subplots()
        #xx = linspace(0, max(temps), 1000)
        #ax.plot(xx, func(xx, *a))
        #ax.scatter(temperatures, func(temperatures, *a), c ='red')
        ax.scatter(temps, mag, alpha = .2)
        setp(ax, **dict(xlabel = 'Temperature', ylabel = '<M>'))
        savefig(f'{targetDirectory}/temp_vs_mag.png')

        #tmp = dict(temps = temps, \
        #temperatures = temperatures, magRange = magRange, mag = mag)
        #IO.savePickle(f'{targetDirectory}/mags.pickle', tmp)
        print(targetDirectory)
