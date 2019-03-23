#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'


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

def create_undirected_tree(z, depth):
    graph = nx.balanced_tree(z, depth)
    return graph

def create_directed_tree(z, depth):
    graph = nx.DiGraph()
    graph = nx.balanced_tree(z, depth, create_using=graph)
    return graph

def create_cayley_tree(z, depth):
    subtrees = [(nx.balanced_tree(z,depth-1), 0) for _ in range(z+1)]
    graph = nx.join(subtrees)
    return graph


def create_erdos_renyi_graph(N, avg_deg=2.):

    p = avg_deg/N

    graph = nx.erdos_renyi_graph(N, p)
    connected_nodes = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(connected_nodes)
    nx.write_gpickle(graph, f'networkData/ER_avgDeg={avg_deg}_N={N}.gpickle', 2)

def create_powerlaw_graph(N, gamma=1.6):
    seq = nx.utils.powerlaw_sequence(N, gamma)
    graph = nx.expected_degree_graph(seq, selfloops = False)
    graph = sorted(nx.connected_component_subgraphs(graph), key=len, reverse=True)[0]
    nx.write_gpickle(graph, f'networkData/scaleFree_gamma={gamma}_N={N}.gpickle')

def create_2D_grid(L):

    graph = nx.grid_2d_graph(L, L, periodic=True)
    nx.write_gpickle(graph, f'networkData/2D_grid_L={L}.gpickle', 2)


def run_mag_series(model, runs, steps, p_initial):

    fig, ax = subplots(figsize=(10,5))
    for i in range(runs):
        model.states = np.random.choice([1,-1], size = model.nNodes, p=[p_initial, 1-p_initial])
        mags = infcy.magTimeSeries(model, burninSamples=int(0), nSamples=int(steps))
        #mags_normal = np.where(np.abs(mags)>0.05, mags, np.nan)
        #mags_switch = np.where(np.abs(mags)<=0.05, mags, np.nan)
        ax.plot(mags)
        #ax.plot(mags_switch, c='r')
    savefig(f'{targetDirectory}/magsTimeSeries.png')
    np.save(f'{targetDirectory}/magsTimeSeries.npy', mags)


def run_mixing(model, num_p_initial, step_size_burnin, num_steps_regress, threshold_regress, num_steps_corr, threshold_corr):

    #mixingTimes = []
    #corrTimes = []
    fig1, ax1 = subplots(figsize=(10,5))
    fig2, ax2 = subplots(figsize=(10,5))
    maxMixingTime = 0
    maxCorrTime = 0
    for prob in tqdm(np.linspace(0.5, 1, num_p_initial)):
        model.states = np.random.choice([-1,1], size = model.nNodes, p=[prob, 1-prob])
        mags, mixingTime, autocorr = infcy.determineMixingTime(model,\
                              stepSizeBurnin = step_size_burnin,\
                              nStepsRegress = int(num_steps_regress),\
                              threshold = threshold_regress,\
                              nStepsCorr = int(num_steps_corr))
        maxMixingTime = max(maxMixingTime, mixingTime)
        corrTime = np.where(np.abs(autocorr) < threshold_corr)[0][0]
        maxCorrTime = max(maxCorrTime, corrTime)
        #func = lambda x, t: np.exp(-x/t)
        #func = lambda x, m, c, c0: c0 + x**m * c
        #a, b = scipy.optimize.curve_fit(func, np.arange(autocorr.size), autocorr, p0=[-1., 1., 0.]) # characteristic autocorrelation time
        #mixingTimes.append(mixingTime)
        #corrTimes.append(corrTime)
        #ax1.plot(autocorr[:corrTime*2])
        #ax1.plot(autocorr)
        #xx = np.linspace(0, 500, 1000)
        #ax1.plot(corrTime, autocorr[corrTime], 'p', c='red')
        ax2.plot(mags)

    print(maxMixingTime)
    print(maxCorrTime)

    maxCorrTime = 0
    for prob in tqdm(np.linspace(0.5, 1, num_p_initial)):
        model.states = np.random.choice([-1,1], size = model.nNodes, p=[prob, 1-prob])
        autocorr = infcy.determineCorrTime(model,\
                              nBurnin = maxMixingTime,\
                              nStepsCorr = int(num_steps_corr))
        corrTime = np.where(np.abs(autocorr) < threshold_corr)[0][0]
        ax1.plot(autocorr)
        ax1.plot(corrTime, autocorr[corrTime], 'p', c='red')
        maxCorrTime = max(maxCorrTime, corrTime)

    #print(mixingTimes)
    print(maxCorrTime)

    fig1.savefig(f'{targetDirectory}/autocorr.png')
    fig2.savefig(f'{targetDirectory}/magTimeSeries.png')

def run_mixing_temps(model, nInitialConfigs, temps, stepSizeBurnin=10, nStepsRegress=int(1e4), thresholdReg=0.05, nStepsCorr=int(1e4), thresholdCorr=0.05):
    mixingTimes, corrTimes = infcy.mixingTimePerTemp(model, \
                            nInitialConfigs, \
                            temps, \
                            stepSizeBurnin, \
                            nStepsRegress, \
                            thresholdReg, \
                            nStepsCorr, \
                            thresholdCorr)
    fig1, ax1 = subplots(figsize=(10,5))
    ax1.plot(temps, mixingTimes, label="mixing time")
    ax1.plot(temps, corrTimes, label= "correlation time")
    ax1.legend()
    fig1.savefig(f'{targetDirectory}/mixing_and_corr_times_perT.png')

    for idx, t in enumerate(temps):
        fig, ax = subplots(figsize=(10,5))
        model.t = t
        model.states = np.random.choice([1,-1], size = model.nNodes, p=[1, 0])
        mags = infcy.magTimeSeries(model, burninSamples=int(0), nSamples=int(1e4))
        #mags_normal = np.where(np.abs(mags)>0.05, mags, np.nan)
        #mags_switch = np.where(np.abs(mags)<=0.05, mags, np.nan)
        fig, ax = subplots(figsize=(10,5))
        ax.plot(mags)
        ax.plot(corrTimes[idx], mags[int(corrTimes[idx])], 'p', c='red')
        fig.savefig(f'{targetDirectory}/mags_T={t}.png')




if __name__ == '__main__':

    now = time.time()
    targetDirectory = f'{os.getcwd()}/Data/{now}'
    os.mkdir(targetDirectory)

    # 2e4 steps with non-single updates and 32x32 grid --> serial-time = parallel-time

    T             = 0.5
    nSamples      = int(1e4) #int(1e6)
    burninSamples = int(1e4) # int(1e6)
    magSide       = '' # which sign should the overall magnetization have (''--> doesn't matter, 'neg' --> flip states if <M> > 0, 'pos' --> flip if <M> < 0)
    updateType    = ''

    #network_path = "networkData/ER_k=3.0_N=500.gpickle"
    network_path = f'{os.getcwd()}/networkData/ER_k=2.5_N=100.gpickle'
    #network_path = "networkData/undirected_tree_z=4_depth=6.gpickle"
    #network_path = "networkData/unweighted_person-person_projection_anonymous_combined_GC_stringToInt.gpickle"
    graph = nx.read_gpickle(network_path)

    #graph = create_undirected_tree(2,6)
    #network_path = 'directed tree z=2, d=5'

    #print(len(graph))


    settings = dict(
        T                = T,\
        nSamples         = nSamples,\
        burninSamples    = burninSamples,\
        updateMethod     = updateType,\
        nNodes           = graph.number_of_nodes(), \
        graph            = network_path
        )
    IO.saveSettings(targetDirectory, settings)

    modelSettings = dict(\
                         graph       = graph,\
                         temperature = T,\
                         updateType  = updateType,\
                         magSide     = magSide
                         )
    model = fastIsing.Ising(**modelSettings)
    updateType = model.updateType


    if len(sys.argv) > 1:
        if sys.argv[1] == "series":
            run_mag_series(model, \
                            runs        = 5,    \
                            steps       = 1e4,  \
                            p_initial   = 0.0)

        if sys.argv[1] == "mixing":
            #run_mixing(model, \
            #                num_p_initial       = 10,   \
            #                step_size_burnin    = 10,   \
            #                num_steps_regress   = 1e4,  \
            #                threshold_regress   = 0.01, \
            #                num_steps_corr      = 5e4,  \
            #                threshold_corr      = 0.05)
            run_mixing_temps(model, 20, np.linspace(0.5, 2, 10))

    else:

        #magRange = array([CHECK]) if isinstance(CHECK, float) else array(CHECK) # ratio of magnetization to be reached
        temps = linspace(0.05, 4, 100)

        mag, sus = infcy.magnetizationParallel(model,       \
                            temps           = temps,        \
                            n               = nSamples,     \
                            burninSamples   = burninSamples)

        fig, ax = subplots(figsize=(10,6))
        ax.scatter(temps, mag, alpha = .2, label='magnetization')
        ax.scatter(temps, sus, alpha = .2, label='susceptibility')
        ax.legend()
        setp(ax, **dict(xlabel = 'Temperature', ylabel = '<M>'))
        savefig(f'{targetDirectory}/temp_vs_mag.png')

        tmp = dict(temps = temps, mag = mag)
        IO.savePickle(f'{targetDirectory}/mags.pickle', tmp)

        np.save(f'{targetDirectory}/mags.npy', mag)
        np.save(f'{targetDirectory}/susceptibility.npy', sus)

    print(targetDirectory)
