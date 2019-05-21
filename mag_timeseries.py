#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'


from Models import fastIsing
from Toolbox import infcy
from Utils import IO
import networkx as nx, itertools, scipy, time, subprocess, \
                os, pickle, sys, argparse, multiprocessing as mp
import itertools
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
from scipy import stats


nthreads = mp.cpu_count()
#nthreads = 1

parser = argparse.ArgumentParser(description='run MC chain and collect system magnetization')
parser.add_argument('T', type=float, help='temperature')
parser.add_argument('dir', type=str, help='target directory')
parser.add_argument('graph', type=str, help='path to pickled graph')
parser.add_argument('--numSamples', type=int, default=1000, help='number of samples per MC run with fixed neighbour states')
parser.add_argument('--abs', action="store_true", help='get absolute magnetization')


if __name__ == '__main__':

    args = parser.parse_args()

    T = args.T
    targetDirectory = args.dir
    os.makedirs(targetDirectory, exist_ok=True)

    # load network
    graph = nx.read_gpickle(args.graph)
    N = len(graph)

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
        #raise Exception('No mixing results found! Please run the mixing script first to determine the mixing time of the model.')
        subprocess.call(['python3', 'LISA_run_mixing.py', f'{args.T}', f'{args.dir}', f'{args.graph}', \
                        '--maxcorrtime', '10000', \
                        '--maxmixing', '10000', \
                        '--corrthreshold', '0.5'])
        mixingResults = IO.loadResults(targetDirectory, 'mixingResults')
        corrTimeSettings = IO.loadResults(targetDirectory, 'corrTimeSettings')
        burninSteps = mixingResults['burninSteps']
        distSamples = mixingResults['distSamples']

    now = time.time()

    abs = 1 if args.abs else 0
    #mag = infcy.magTimeSeries(model, burninSteps, args.numSamples, abs=abs)
    #np.save(f'{targetDirectory}/magTimeSeries_T={args.T}_NSteps_{args.numSamples}_{now}.npy', mag)
    mag = model.magTimeSeries(args.numSamples, burninSteps)
    np.save(f'{targetDirectory}/magTimeSeries_T={args.T}_nSteps={args.numSamples}_{now}.npy', mag)
