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

nthreads = mp.cpu_count() - 1
#nthreads = 1

parser = argparse.ArgumentParser(description='run MC chain and compute MI based on the joint PDF of the central node and its neighbours')
parser.add_argument('T', type=float, help='temperature')
parser.add_argument('dir', type=str, help='target directory')
parser.add_argument('graph', type=str, help='path to pickled graph')
parser.add_argument('nodes', type=str, help='path to numpy array of node IDs')
parser.add_argument('--neighboursDir', type=str, default='', help='path to directory containing pickled neighbours dictionary')
parser.add_argument('--maxDist', type=int, default=-1, help='max distance to central node')
parser.add_argument('--runs', type=int, default=1, help='number of repetititve runs')
parser.add_argument('--bins', type=int, default=10, help='number of bins for average magnetization of neighbours')
parser.add_argument('--repeats', type=int, default=10, help='number of parallel MC runs used to estimate MI')
parser.add_argument('--numSteps', type=int, default=1000, help='number of system updates')
parser.add_argument('--threshold', type=float, default=0.1, help='threshold to determine mag switching')


if __name__ == '__main__':

    print("starting with average neighbour MI approach")

    start = timer()

    args = parser.parse_args()

    T = args.T
    targetDirectory = args.dir
    os.makedirs(targetDirectory, exist_ok=True)

    graph = nx.read_gpickle(args.graph)
    N = len(graph)

    # load network
    if args.maxDist > 0:
        maxDist = args.maxDist
    else:
        maxDist = nx.diameter(graph)

    nodes = np.load(args.nodes)

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

    try:
        mixingResults = IO.loadResults(targetDirectory, 'mixingResults')
        corrTimeSettings = IO.loadResults(targetDirectory, 'corrTimeSettings')
        burninSteps = mixingResults['burninSteps']
        distSamples = mixingResults['distSamples']
        magLevel    = mixingResults['magLevel']
        print(f'mixing time = {burninSteps}')
        print(f'correlation time = {distSamples}')
        print(f'absolute average magnetization = {magLevel}')

    except:
        #raise Exception('No mixing results found! Please run the mixing script first to determine the mixing and correlation time of the model.')
        subprocess.call(['python3', 'LISA_run_mixing.py', f'{args.T}', f'{args.dir}', f'{args.graph}', \
                        '--maxcorrtime', '10000', \
                        '--maxmixing', '10000', \
                        '--corrthreshold', '0.5'])
        mixingResults = IO.loadResults(targetDirectory, 'mixingResults')
        corrTimeSettings = IO.loadResults(targetDirectory, 'corrTimeSettings')
        burninSteps = mixingResults['burninSteps']
        distSamples = mixingResults['distSamples']
        magLevel    = mixingResults['magLevel']

    try:
        if len(args.neighboursDir) > 0:
            neighboursG = IO.loadPickle(args.neighboursDir, 'neighboursG')
        else:
            neighboursG = IO.loadPickle(targetDirectory, 'neighboursG')

    except:
        print(f'determining neighbours')
        neighboursG = model.neighboursAtDistAllNodes(nodes, maxDist)
        IO.savePickle(targetDirectory, 'neighboursG', neighboursG)



    snapshotSettingsJoint = dict( \
        nSteps    = args.numSteps, \
        repeats     = args.repeats, \
        burninSamples = burninSteps, \
        maxDist     = maxDist, \
        nBins       = args.bins, \
        threshold   = args.threshold
    )
    IO.saveSettings(targetDirectory, snapshotSettingsJoint, 'jointSnapshots')


    for r in range(args.runs):

        avgSnapshotsPos, avgSnapshotsNeg, avgSnapshotsSwitch, Z, mags = infcy.getSystemStates(model, nodes, \
                                                                            neighboursG, \
                                                                            **snapshotSettingsJoint, threads=nthreads)

        now = time.time()

        MI_avg_pos, _, HX_pos = infcy.processJointSnapshotsNodes(avgSnapshotsPos, Z[0], nodes, maxDist)
        MI_avg_neg, _, HX_neg = infcy.processJointSnapshotsNodes(avgSnapshotsNeg, Z[1], nodes, maxDist)
        MI_avg_switch, _, HX_switch = infcy.processJointSnapshotsNodes(avgSnapshotsSwitch, Z[2], nodes, maxDist)

        IO.savePickle(targetDirectory, f'MI_pos_meanField_nodes_{now}', MI_avg_pos)
        IO.savePickle(targetDirectory, f'HX_pos_meanField_nodes_{now}', HX_pos)
        IO.savePickle(targetDirectory, f'MI_neg_meanField_nodes_{now}', MI_avg_neg)
        IO.savePickle(targetDirectory, f'HX_neg_meanField_nodes_{now}', HX_neg)
        IO.savePickle(targetDirectory, f'MI_switch_meanField_nodes_{now}', MI_avg_switch)
        IO.savePickle(targetDirectory, f'HX_switch_meanField_nodes_{now}', HX_switch)


    print(f'time elapsed: {timer()-start : .2f} seconds')

    print(targetDirectory)
