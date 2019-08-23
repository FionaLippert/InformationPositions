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


def compute_entropies(snapshots, nSamples):
    sum = np.sum([np.sum(np.fromiter(s.values(), dtype=int)) for s in snapshots.values()])
    #print(sum)
    condEntropies = [infoTheory.entropyEstimateH2(np.fromiter(s.values(), dtype=int)) for s in snapshots.values()]
    #print(condEntropies)
    condH = np.sum([condEntropies[i] * np.sum(np.fromiter(s.values(), dtype=int))/(nSamples) for i, s in enumerate(snapshots.values())])
    #print(f'H2(S|s_i) = {condH}')

    allSnapshots = {}
    for _, s in snapshots.items():
        for k, v in s.items():
            if k in allSnapshots.keys():
                allSnapshots[k] += v
            else:
                allSnapshots[k] = v
    systemH = infoTheory.entropyEstimateH2(np.fromiter(allSnapshots.values(), dtype=int))
    #print(f'H2(S) = {systemH}')

    return condH, systemH



parser = argparse.ArgumentParser(description='')
parser.add_argument('T', type=float, help='temperature')
parser.add_argument('dir', type=str, help='target directory')
parser.add_argument('graph', type=str, help='path to pickled graph')
parser.add_argument('--excludeNodes', action="store_true", help='exclude fixed nodes from system entropy')
parser.add_argument('--onlyRandom', action="store_true", help='do not run greedy algorithm, only random k-sets')
parser.add_argument('--bruteForce', action="store_true", help='do not run greedy algorithm, try all possible k-sets')
parser.add_argument('--heuristic', type=str, default='', help='construct greedy sets based on IV heuristic and MI-radius')
parser.add_argument('--trials', type=int, default=1, help='number of trials. The median of all MI estimates is saved')
parser.add_argument('--snapshots', type=int, default=10000, help='number of system snapshots')
parser.add_argument('--magSide', type=str, default='', help='fix magnetization to one side (pos/neg)')
parser.add_argument('--initState', type=int, default=1, help='initial system state')
parser.add_argument('--k_max', type=int, default=3, help='max k-set size to be considered')
parser.add_argument('--k_min', type=int, default=1, help='min k-set size to be considered (only for heuristic approach)')

nthreads = mp.cpu_count()

if __name__ == '__main__':

    start = timer()

    args = parser.parse_args()

    targetDirectory = args.dir
    os.makedirs(targetDirectory, exist_ok=True)

    # load data
    graph = nx.read_gpickle(args.graph)
    N = len(graph)
    T = args.T

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

    try:
        mixingResults = IO.loadResults(targetDirectory, 'mixingResults')
        corrTimeSettings = IO.loadResults(targetDirectory, 'corrTimeSettings')
        burninSteps = mixingResults['burninSteps']
        distSamples = mixingResults['distSamples']

    except:
        #raise Exception('No mixing results found! Please run the mixing script first to determine the mixing time of the model.')
        subprocess.call(['python3', 'run_mixing.py', f'{T}', f'{args.dir}', f'{args.graph}', \
                        '--maxcorrtime', '10000', \
                        '--maxmixing', '10000', \
                        '--corrthreshold', '0.5'])
        mixingResults = IO.loadResults(targetDirectory, 'mixingResults')
        corrTimeSettings = IO.loadResults(targetDirectory, 'corrTimeSettings')
        burninSteps = mixingResults['burninSteps']
        distSamples = mixingResults['distSamples']


    systemSnapshotSettings = dict( \
        nSnapshots    = args.snapshots, \
        burninSamples = int(burninSteps), \
        distSamples     = int(distSamples)
    )
    IO.saveSettings(targetDirectory, systemSnapshotSettings, 'systemSnapshots')



    selected_nodes = []
    remaining_nodes = list(graph)
    nodes_array = np.array(remaining_nodes)

    mi_greedy = {}
    h_greedy = {}

    mi_random = {}
    h_random = {}

    mi_brute_force = {}
    h_brute_force = {}

    if args.bruteForce:

        sets = [list(s) for s in itertools.combinations(nodes_array, args.k_max)]
        print(sets)
        if args.excludeNodes:
            systemNodes = [list(nodes_array[~np.isin(nodes_array, s)].astype(int)) for s in sets]
        else:
            systemNodes = [list(nodes_array) for s in sets]

        snapshots = simulation.getSystemSnapshotsSets(model, systemNodes, sets, \
                      **systemSnapshotSettings, threads = nthreads, initStateIdx = args.initState)

        for i, s in enumerate(sets):
            condRand, systemRand = infoTheory.compute_entropies(snapshots[i], args.snapshots)
            mi_brute_force[tuple(s)] = systemRand - condRand
            h_brute_force[tuple(s)] = condRand

    if args.onlyRandom:

        sets = [list(np.random.choice(nodes_array, k, replace=False)) for k in range(1, args.k_max + 1)]
        print(sets)
        if args.excludeNodes:
            systemNodes = [list(nodes_array[~np.isin(nodes_array, s)].astype(int)) for s in sets]
        else:
            systemNodes = [list(nodes_array) for s in sets]

        snapshots = simulation.getSystemSnapshotsSets(model, systemNodes, sets, \
                      **systemSnapshotSettings, threads = nthreads, initStateIdx = args.initState)

        for i, s in enumerate(sets):
            condRand, systemRand = infoTheory.compute_entropies(snapshots[i], args.snapshots)
            mi_random[tuple(s)] = systemRand - condRand
            h_random[tuple(s)] = condRand

    elif len(args.heuristic) > 0:

        MI_avg = IO.SimulationResult.loadNewestFromPickle(args.heuristic, 'avg').mi

        # normalize IV
        max_IV = np.max([np.nansum(MI_avg[n]) for n in MI_avg.keys()])
        min_IV = np.min([np.nansum(MI_avg[n]) for n in MI_avg.keys()])
        IV = { n : (np.nansum(mi)-min_IV)/(max_IV-min_IV) for n, mi in MI_avg.items() }

        def get_dmax(mi, p):
            idx = np.where(mi < mi[0]*p)[0]
            if idx.size > 0:
                return idx[0]
            else:
                return mi.size

        def get_dmax_thr(mi, mi_max, thr=0.1):
            #idx = np.where(mi < mi[0]*p)[0]
            idx = np.where(mi/mi_max < thr)[0]
            if idx.size > 0:
                return idx[0]
            else:
                return mi.size

        def overlap_node_set(G, node, node_set, mi, ratio=True, p=0.5):
            #d_max = {n : get_dmax_thr(mi[n], p) for n in node_set}
            mi_max = np.max([vals[0] for vals in mi.values()])
            d_max = {n : get_dmax_thr(mi[n], mi_max) for n in node_set}

            set1 = set(list(nx.ego_graph(G, node, get_dmax_thr(mi[node], mi_max))))
            set2 = set()
            for n in node_set:
                set2 = set2.union(list(nx.ego_graph(G, n, d_max[n])))
            #overlap = len(set1.intersection(set2)) / len(set(set1).union(set2)) if ratio else len(set1.intersection(set2))
            overlap = len(set1.intersection(set2)) / len(set1)
            #overlap = len(set1 - set2) / len(G)
            return overlap

        def covering_node_set(G, node_set, mi, ratio=True, thr=0.1):
            mi_max = np.max([vals[0] for vals in mi.values()])
            d_max = {n : get_dmax_thr(mi[n], mi_max, thr) for n in node_set}

            cover = set()
            for n in node_set:
                cover = cover.union(list(nx.ego_graph(G, n, d_max[n])))

            return len(cover)/len(G)


        ranking = sorted(IV.items(), key=lambda kv: kv[1], reverse=True)
        ranked_nodes, iv_values = zip(*ranking)

        #TODO greedily select next node that minimizes overlap and maximizes IV

        sets = {}
        systemNodes = []

        #selected_nodes = [ranked_nodes[0]]
        #remaining_nodes.remove(ranked_nodes[0])

        pool = ranked_nodes[:int(N/2)]

        mi_max = np.max([vals[0] for vals in MI_avg.values()])
        d_max = {n : get_dmax_thr(MI_avg[n], mi_max) for n in graph}
        print(d_max.values())

        print([d_max[n] for n in ranked_nodes[int(N/2):]])

        for k in range(args.k_min, args.k_max + 1):

            print(k)

            max_score = -np.infty
            """
            top_set = []
            for i, s in enumerate(itertools.combinations(pool, k)):
                overlap = np.mean([ overlap_node_set(graph, n, list(set(s)-set([n])), MI_avg) for n in s])
                #covering = covering_node_set(graph, s, MI_avg, thr=0.01)
                iv = np.mean([ IV[n] for n in s])
                score = iv/(overlap + 0.001)
                #score = iv * (1-overlap)
                #score = iv - overlap
                #score = iv * covering

                #print(f'--------------- node set {s}, overlap = {overlap}, iv = {iv} ---------------')
                if score > max_score:
                    top_set = s
                    max_score = score
                    max_iv = iv
                    #max_covering = covering
            #print(f'k={k}: score = {max_score}, iv = {max_iv}, covering = {max_covering}')
            sets[k] = top_set
            """

            used_nodes = set(sets[k-1]) if k > 1 else set()
            for i, s in enumerate(set(remaining_nodes)-used_nodes):
                #candidate_set = [s] + sets[k-1]
                #overlap = np.mean([ overlap_node_set(graph, n, list(set(candidate_set)-set([n])), MI_avg) for n in candidate_set)
                overlap = overlap_node_set(graph, s, used_nodes, MI_avg)
                iv = IV[s]
                score = iv/(overlap + 0.001)
                score = iv
                #score = iv * overlap
                #score = iv - overlap

                #print(f'--------------- node set {s}, overlap = {overlap}, iv = {iv} ---------------')
                if score > max_score:
                    sets[k] = list(used_nodes) + [s]
                    max_score = score

            #print(i)




            #for i, n in enumerate(remaining_nodes):
            #    scores[i] = IV[n] #
            #    scores[i] = - overlap_node_set(graph, n, selected_nodes, MI_avg)

            #top_idx = np.argmax(scores)
            #top_node = remaining_nodes[top_idx]

            #print(top_node, IV[top_node], overlap_node_set(graph, top_node, selected_nodes, MI_avg))

            #selected_nodes.append(top_node)
            #remaining_nodes.remove(top_node)

            #sets += selected_nodes
            #if args.excludeNodes:
            #    systemNodes += (remaining_nodes)
            #else:
            #    systemNodes += (list(graph))

        #sets = [ selected_nodes[:k] for k in range(1, args.k_max + 1) ]

        #sets_random = [ list(np.random.choice(nodes_array, k, replace=False)) for k in range(1, args.k_max + 1) ]

        sets = list(sets.values())

        if args.excludeNodes:
            systemNodes = [list(nodes_array[~np.isin(nodes_array, s)].astype(int)) for s in sets]
            #systemNodes_random = [list(nodes_array[~np.isin(nodes_array, s)].astype(int)) for s in sets_random]
        else:
            systemNodes = [list(nodes_array) for s in sets]
            #systemNodes_random = [list(nodes_array) for s in sets]

        snapshots = simulation.getSystemSnapshotsSets(model, systemNodes, sets, \
                      **systemSnapshotSettings, threads = nthreads, initStateIdx = args.initState)

        for i, s in enumerate(sets):
            print(f'--------------- node set {s} ---------------')
            #print(systemNodes[i])

            condH, systemH = infoTheory.compute_entropies(snapshots[i], args.snapshots)
            mi_greedy[tuple(s)] = systemH - condH
            h_greedy[tuple(s)]  = condH
            print(f'MI = {systemH - condH}')
        """
        snapshots = simulation.getSystemSnapshotsSets(model, systemNodes_random, sets_random, \
                      **systemSnapshotSettings, threads = nthreads, initStateIdx = args.initState)

        for i, s in enumerate(sets_random):
            print(f'--------------- node set {s} ---------------')
            #print(systemNodes[i])

            condH, systemH = infoTheory.compute_entropies(snapshots[i], args.snapshots)
            #mi_greedy[tuple(s)] = systemH - condH
            #h_greedy[tuple(s)]  = condH
            print(f'MI = {systemH - condH}')
        """

    else:

        for k in range(1, args.k_max + 1):

            sets = [ selected_nodes + [n] for n in remaining_nodes ]
            print(sets)

            sets.append(list(np.random.choice(nodes_array, k, replace=False)))

            if args.excludeNodes:
                systemNodes = [list(nodes_array[~np.isin(nodes_array, s)].astype(int)) for s in sets]
            else:
                systemNodes = [list(nodes_array) for s in sets]

            MI      = np.zeros(len(remaining_nodes))
            condH   = np.zeros(len(remaining_nodes))
            systemH = np.zeros(len(remaining_nodes))

            #print(f'T = {model.temperature}, magSide = {model.magSide}')
            snapshots = simulation.getSystemSnapshotsSets(model, systemNodes, sets, \
                          **systemSnapshotSettings, threads = nthreads, initStateIdx = args.initState)

            #IO.savePickle(targetDirectory, f'backup_snapshots_k={k}', snapshots)

            for i, n in enumerate(remaining_nodes):
                print(f'--------------- node set {selected_nodes + [n]} ---------------')

                condH[i], systemH[i] = infoTheory.compute_entropies(snapshots[i], args.snapshots)
                MI[i] = systemH[i] - condH[i]
                print(f'MI = {MI[i]}')


            ranking = np.argsort(MI)
            top_idx = ranking[-1]
            top_node = remaining_nodes[top_idx]

            print(f'best node choice: {top_node} with MI = {MI[top_idx]}')

            selected_nodes.append(top_node)
            remaining_nodes.remove(top_node)

            mi_greedy[tuple(selected_nodes)] = MI[top_idx]
            h_greedy[tuple(selected_nodes)] = condH[top_idx]

            condRand, systemRand = infoTheory.compute_entropies(snapshots[-1], args.snapshots)
            mi_random[tuple(sets[-1])] = systemRand - condRand
            h_random[tuple(sets[-1])] = condRand


    result = IO.SimulationResult('greedy', \
                networkSettings     = networkSettings, \
                modelSettings       = modelSettings, \
                snapshotSettings    = systemSnapshotSettings, \
                corrTimeSettings    = corrTimeSettings, \
                mixingResults       = mixingResults, \
                miGreedy            = mi_greedy, \
                hCondGreedy         = h_greedy, \
                miRandom            = mi_random, \
                hCondRandom         = h_random, \
                miBruteForce        = mi_brute_force, \
                hCondBruteForce     = h_brute_force, \
                computeTime         = timer()-start )
    result.saveToPickle(targetDirectory)
