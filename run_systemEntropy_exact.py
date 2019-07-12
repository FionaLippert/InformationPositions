#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'


from Models import fastIsing
from Toolbox import infoTheory, simulation
from Utils import IO
import networkx as nx, itertools, scipy, time, subprocess, \
                os, pickle, sys, argparse, multiprocessing as mp
import itertools
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
from scipy import stats

def hamiltonian(G, mapping, config):
    h = 0
    #print(G.edges())
    for n1, n2 in G.edges():
        h -= config[mapping[n1]] * config[mapping[n2]]
    return h

def exact_entropy(G, mapping, beta, agent_states=[-1, 1]):
    all_h = np.zeros(len(agent_states)**len(G))

    for i, config in enumerate(itertools.product(agent_states, repeat=len(G))):
        all_h[i] = hamiltonian(G, mapping, config)

    all_p = np.exp(-beta * all_h)
    Z = np.sum(all_p)
    all_p = all_p / Z

    H_S = stats.entropy(all_p, base=2)
    return H_S


def compute_entropies(snapshots, nSamples):
    sum = np.sum([np.sum(np.fromiter(s.values(), dtype=int)) for s in snapshots.values()])
    print(sum)
    condEntropies = [infoTheory.entropyEstimateH2(np.fromiter(s.values(), dtype=int)) for s in snapshots.values()]
    print(condEntropies)
    condH = np.sum([condEntropies[i] * np.sum(np.fromiter(s.values(), dtype=int))/(nSamples) for i, s in enumerate(snapshots.values())])
    print(f'H2(S|s_i) = {condH}')

    allSnapshots = {}
    for _, s in snapshots.items():
        for k, v in s.items():
            if k in allSnapshots.keys():
                allSnapshots[k] += v
            else:
                allSnapshots[k] = v
    systemH = infoTheory.entropyEstimateH2(np.fromiter(allSnapshots.values(), dtype=int))

    print(f'H2(S) = {systemH}')

    return condH, systemH


nthreads = mp.cpu_count()
#nthreads = 1


parser = argparse.ArgumentParser(description='run MC chain, sample system snapshots and estimate the system entropy conditioned on the given node set')
parser.add_argument('T', type=float, help='temperature')
parser.add_argument('dir', type=str, help='target directory')
parser.add_argument('graph', type=str, help='path to pickled graph')
parser.add_argument('--trials', type=int, default=1, help='number of trials')
parser.add_argument('--minSnapshots', type=int, default=10, help='min number of system snapshots')
parser.add_argument('--maxSnapshots', type=int, default=1000, help='min number of system snapshots')
parser.add_argument('--stepsSnapshots', type=int, default=10, help='number of steps between min and max number of system snapshots')
#parser.add_argument('--repeats', type=int, default=10, help='number of parallel MC runs used to estimate entropy')
parser.add_argument('--magSide', type=str, default='', help='fix magnetization to one side (pos/neg)')
parser.add_argument('--initState', type=int, default=-1, help='initial system state')



if __name__ == '__main__':

    start = timer()

    args = parser.parse_args()

    T = args.T
    targetDirectory = args.dir
    os.makedirs(targetDirectory, exist_ok=True)

    # load network
    graph = nx.read_gpickle(args.graph)
    N = len(graph)


    networkSettings = dict( \
        path = args.graph, \
        nNodes = N
    )

    # setup Ising model with nNodes spin flip attempts per simulation step
    modelSettings = dict( \
        temperature     = T, \
        updateType      = 'async' ,\
        magSide         = args.magSide if args.magSide in ['pos', 'neg'] else ''
    )
    #IO.saveSettings(targetDirectory, modelSettings, 'model')
    model = fastIsing.Ising(graph, **modelSettings)

    #print(model.mapping)


    print('------------- compute exact system entropy --------------------')

    H_S = exact_entropy(graph, model.mapping, 1/float(T))
    print(f'exact entropy = {H_S}')

    print('--------- estimate system entropy from simulation -------------')

    try:
        assert False
        mixingResults = IO.loadResults(targetDirectory, 'mixingResults')
        corrTimeSettings = IO.loadResults(targetDirectory, 'corrTimeSettings')
        burninSteps = mixingResults['burninSteps']
        distSamples = mixingResults['distSamples']

    except:
        #raise Exception('No mixing results found! Please run the mixing script first to determine the mixing time of the model.')
        subprocess.call(['python3', 'run_mixing.py', f'{args.T}', f'{args.dir}', f'{args.graph}', \
                        '--maxcorrtime', '10000', \
                        '--maxmixing', '10000', \
                        '--corrthreshold', '0.5'])
        mixingResults = IO.loadResults(targetDirectory, 'mixingResults')
        corrTimeSettings = IO.loadResults(targetDirectory, 'corrTimeSettings')
        burninSteps = mixingResults['burninSteps']
        distSamples = mixingResults['distSamples']


    now = time.time()
    start = timer()

    sRange = np.linspace(args.minSnapshots, args.maxSnapshots, args.stepsSnapshots)

    allH2 = np.zeros((args.stepsSnapshots, args.trials))
    allNaive = np.zeros((args.stepsSnapshots, args.trials))

    for i, s in enumerate(sRange):

        systemSnapshotSettings = dict( \
            nSnapshots    = s, \
            burninSamples = int(burninSteps), \
            distSamples     = int(distSamples)
        )

        for t in range(args.trials):

            snapshots = simulation.getSystemSnapshots(model, np.array(list(graph)), **systemSnapshotSettings, \
                               threads = nthreads, initStateIdx = args.initState)

            H2 = infoTheory.entropyEstimateH2(np.fromiter(snapshots.values(), dtype=int))

            p = np.array(np.fromiter(snapshots.values(), dtype=float))
            p = p / np.sum(p)
            naive = stats.entropy(p, base=2)

            allH2[i, t] = H2
            allNaive[i, t] = naive

        print(f'{s} snapshots: corrected entropy = {np.mean(allH2[i])}, naive entropy = {np.mean(allNaive[i])}')

    result = IO.SimulationResult('system', \
                networkSettings     = networkSettings, \
                modelSettings       = modelSettings, \
                corrTimeSettings    = corrTimeSettings, \
                mixingResults       = mixingResults, \
                hx                  = allH2, \
                hxNaive             = allNaive, \
                hxExact             = H_S, \
                snapshotRange       = sRange, \
                computeTime         = timer()-start )
    result.saveToPickle(targetDirectory)



    print(f'time elapsed: {timer()-start : .2f} seconds')
