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


parser = argparse.ArgumentParser(description='determine mixing and correlation time')
parser.add_argument('T', type=float, help='temperature')
parser.add_argument('dir', type=str, help='target directory')
parser.add_argument('graph', type=str, help='path to pickled graph')
parser.add_argument('--maxcorrtime', type=int, default=5000, help='max correlation time to be used as sample distance')
parser.add_argument('--maxmixing', type=int, default=5000, help='max mixing time to be used for burn-in samples')




if __name__ == '__main__':

    start = timer()

    args = parser.parse_args()

    T = args.T #float(sys.argv[1])
    targetDirectory = args.dir #sys.argv[2]

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
        nInitialConfigs = 10, \
        burninSteps  = 10, \
        nStepsRegress   = int(1e3), \
        nStepsCorr      = int(1e4), \
        thresholdReg    = 0.1, \
        thresholdCorr   = 0.01
    )
    IO.saveSettings(targetDirectory, mixingTimeSettings, 'mixingTime')
    mixingTime, meanMag, corrTime, mags = infcy.determineCorrTime(model, **mixingTimeSettings)
    print(f'correlation time = {corrTime}')
    print(f'mixing time      = {mixingTime}')
    print(f'mag level        = {meanMag}')

    burninSteps = min(mixingTime, args.maxmixing)
    distSamples = min(corrTime, args.maxcorrtime)

    mixingResults = dict(\
        mixingTime = mixingTime, \
        burninSteps = burninSteps, \
        corrTime = corrTime, \
        distSamples = distSamples, \
        magLevel = meanMag
    )
    IO.saveResults(targetDirectory, mixingResults, 'mixingResults')

    #for key, values in mags.items():
    #    np.save(f'{targetDirectory}/magSeries_{key}.npy', np.array(values))


    corrTimeSettings = dict( \
        nInitialConfigs = 10, \
        burninSteps  = burninSteps, \
        nStepsCorr      = int(1e4), \
        thresholdCorr   = 0.01, \
        checkMixing     = 0
    )
    IO.saveSettings(targetDirectory, corrTimeSettings, 'corrTime')
    

    print(f'time elapsed: {timer()-start : .2f} seconds')

    print(targetDirectory)
