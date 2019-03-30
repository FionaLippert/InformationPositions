#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'


from Models import fastIsing
from Toolbox import infcy
from Utils import IO
import networkx as nx, itertools, scipy, time, \
                os, pickle, sys, argparse, multiprocessing as mp
import itertools
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
from scipy import stats

nthreads = mp.cpu_count() - 1
#nthreads = 1


parser = argparse.ArgumentParser(description='run MC chain and compute MI based on conditional PDF of the central node with neighbour states fixed')
parser.add_argument('T', type=float, help='temperature')
parser.add_argument('dir', type=str, help='target directory')
parser.add_argument('graph', type=str, help='path to pickled graph')
parser.add_argument('nodes', type=str, help='path to numpy array of node IDs')
parser.add_argument('maxDist', type=int, help='max distance to central node')
parser.add_argument('--runs', type=int, default=1, help='number of repetitive runs')



if __name__ == '__main__':

    start = timer()

    args = parser.parse_args()

    T = args.T
    targetDirectory = args.dir
    maxDist = args.maxDist

    # load network
    #z = 2
    #maxDist = 8
    #subtrees = [(nx.balanced_tree(z,maxDist-1), 0) for _ in range(z+1)]
    #graph = nx.join(subtrees)
    #graph = nx.balanced_tree(z, maxDist)
    graph = nx.read_gpickle(args.graph)
    N = len(graph)
    nodes = np.load(args.nodes)
    #degrees = [graph.degree[n] for n in nodes]

    #path = f'nx.balanced_tree({z},{maxDist})'
    #path = 'nx.path_graph'

    networkSettings = dict( \
        path = args.graph, \
        nNodes = N, \
        nodes = args.nodes
    )


    # setup Ising model with nNodes spin flip attempts per simulation step
    modelSettings = dict( \
        temperature     = T, \
        updateType      = 'async' ,\
        magSide         = ''
    )
    IO.saveSettings(targetDirectory, modelSettings, 'model')
    model = fastIsing.Ising(graph, **modelSettings)

    try:
        mixingResults = IO.loadResults(targetDirectory, 'mixingResults')
        corrTimeSettings = IO.loadResults(targetDirectory, 'corrTimeSettings')
        burninSteps = mixingResults['burninSteps']
        distSamples = mixingResults['distSamples']

    except:
        raise Exception('No mixing results found! Please run the mixing script first to determine the mixing and correlation time of the model.')

        """
        # determine mixing/correlation time
        mixingTimeSettings = dict( \
            nInitialConfigs = 10, \
            burninSteps  = 10, \
            nStepsRegress   = int(1e3), \
            nStepsCorr      = int(1e4), \
            thresholdReg    = 0.05, \
            thresholdCorr   = 0.01
        )
        IO.saveSettings(targetDirectory, mixingTimeSettings, 'mixingTime')
        mixingTime, meanMag, distSamples, mags = infcy.determineCorrTime(model, **mixingTimeSettings)
        print(f'correlation time = {distSamples}')
        print(f'mixing time      = {mixingTime}')
        print(f'mag level        = {meanMag}')
    

        mixingResults = dict(\
            mixingTime = mixingTime, \
            corrTime = distSamples, \
            magLevel = meanMag
        )
        IO.saveResults(targetDirectory, mixingResults, 'mixingResults')

        mixingTime = min(mixingTime, 5000)


        corrTimeSettings = dict( \
            nInitialConfigs = 10, \
            burninSteps  = mixingTime, \
            nStepsCorr      = int(1e4), \
            thresholdCorr   = 0.01, \
            checkMixing     = 0
        )
        IO.saveSettings(targetDirectory, corrTimeSettings, 'corrTime')
        """


    #allNeighbours_G, allNeighbours_idx = model.neighboursAtDist(model.mapping[node], maxDist)
    
    
    pairwiseMISettings = dict( \
        repeats    = 10, \
        burninSamples = burninSteps, \
        nSamples     = int(1e3), \
        distSamples   = distSamples, \
        distMax = maxDist
    )
    IO.saveSettings(targetDirectory, pairwiseMISettings, 'pairwise')

    #result_dir = f'{targetDirectory}/MI_pairwise'
    #if not os.path.isdir(result_dir): os.mkdir(result_dir)
    
    for i in range(args.runs):
        _, MI, corr, degrees = infcy.runMI(model, nodes = nodes, **pairwiseMISettings)
        MIs_pairwise = np.array([np.nanmean(MI[i,:,:], axis=1) for i in range(MI.shape[0])])
        print(MIs_pairwise)
        #np.save(f'{targetDirectory}/MI_pairwise_T={model.t}.npy', MIs_pairwise[0])

        now = time.time()
        np.save(f'{targetDirectory}/MI_pairwise_{now}.npy', MI)
        np.save(f'{targetDirectory}/corr_pairwise_{now}.npy', corr)
    
    print(f'time elapsed: {timer()-start : .2f} seconds')

    print(targetDirectory)
