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


parser = argparse.ArgumentParser(description='run MC chain and collect system states')
parser.add_argument('T', type=float, help='temperature')
parser.add_argument('dir', type=str, help='target directory')
parser.add_argument('graph', type=str, help='path to pickled graph')
parser.add_argument('--nSamples', type=int, default=1000, help='number of system state samples')




if __name__ == '__main__':

    start = timer()

    args = parser.parse_args()

    T = args.T #float(sys.argv[1])
    targetDirectory = args.dir #sys.argv[2]
    os.makedirs(targetDirectory, exist_ok=True)

    # load network
    graph = nx.read_gpickle(args.graph)
    N = len(graph)

    networkSettings = dict( \
        path = args.graph, \
        nNodes = N
    )
    IO.saveSettings(targetDirectory, networkSettings, 'network')


    # setup Ising model with nNodes spin flip attempts per simulation step
    modelSettings = dict( \
        temperature     = T, \
        updateType      = 'async' ,\
        magSide         = ''
    )
    IO.saveSettings(targetDirectory, modelSettings, 'model')
    model = fastIsing.Ising(graph, **modelSettings)


    # determine mixing/correlation time
    mixingTimeSettings = dict( \
        nInitialConfigs = 50, \
        burninSteps  = 10, \
        nStepsRegress   = int(1e3), \
        nStepsCorr      = int(1e4), \
        thresholdReg    = 0.1, \
        thresholdCorr   = 0.1
    )
    IO.saveSettings(targetDirectory, mixingTimeSettings, 'mixingTime')
    mixingTime, meanMag, corrTime, mags = infcy.determineCorrTime(model, **mixingTimeSettings)
    print(f'correlation time = {corrTime}')
    print(f'mixing time      = {mixingTime}')
    print(f'mag level        = {meanMag}')

    burninSteps = min(mixingTime, 5000)

    mixingResults = dict(\
        mixingTime = mixingTime, \
        burninSteps = burninSteps, \
        corrTime = corrTime, \
        distSamples = corrTime, \
        magLevel = meanMag
    )
    IO.saveResults(targetDirectory, mixingResults, 'mixingResults')

    states = infcy.simulateGetStates(model, burninSteps=burninSteps, nSamples = args.nSamples)

    np.save(os.path.join(targetDirectory), 'system_states.npy', states)
    with open(os.path.join(targetDirectory, f'node_mapping.pickle'), 'wb') as f:
        pickle.dump(model.mapping, f, protocol=pickle.HIGHEST_PROTOCOL)


    print(f'time elapsed: {timer()-start : .2f} seconds')

    print(targetDirectory)
