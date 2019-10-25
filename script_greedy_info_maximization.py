#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

import subprocess, re, os, glob, itertools
import numpy as np
import networkx as nx
from Utils import IO

def get_timestamp(path):
    no_ext = os.path.splitext(path)[0]
    timestamp = no_ext.split('_')[-1]
    return timestamp

N=50
gtype = f'small_graphs/N={N}_p=0.05'
#vs = [1] #list(range(6,10))
vs = list(range(10))

for v in vs:
    gname = f'ER_k={0.05*1.2*N:.2f}_N={N}_v{v}'

    T_result = IO.TempsResult.loadFromPickle(f'tempsData/{gtype}/{gname}', f'{gname}_tempsResults.pickle')

    #Ts = [T_result.T_o, T_result.T_c, T_result.T_d]
    #magSides = ['pos', 'fair', 'fair']


    temps = [T_result.T_o]
    temps_str = ['T_o']
    magSides = ['pos']

    """
    for j, T in enumerate(Ts):

        for k in range(1, 4):
            subprocess.call(['python3', 'run_greedy_info_maximization.py', \
                 f'{T:.2f}', \
                 f'output_systemEntropyGreedy/{gtype}/{graph}/T={T:.2f}/brute_force/k={k}', \
                 f'networkData/{gtype}/{graph}/{graph}.gpickle', \
                 '--k_max', str(k), \
                 '--trials', '1', \
                 '--snapshots', '100000', \
                 '--excludeNodes', \
                 '--bruteForce', \
                 '--magSide', magSides[j]])


    for j, T in enumerate(Ts):
        for i in range(10):
            subprocess.call(['python3', 'run_greedy_info_maximization.py', \
                 f'{T:.2f}', \
                 #f'output_systemEntropyGreedy/{gtype}/{graph}/T={T:.2f}/brute_force', \
                 f'output_systemEntropyGreedy/{gtype}/{graph}/T={T:.2f}/10_trials_nodesExcluded', \
                 f'networkData/{gtype}/{graph}/{graph}.gpickle', \
                 '--k_max', str(int(N/2)), \
                 #'--k_max', '3', \
                 '--trials', '1', \
                 '--snapshots', '100000', \
                 '--excludeNodes', \
                 #'--bruteForce', \
                 '--magSide', magSides[j]])

    """
    for j, (T_val, T_str) in enumerate(zip(temps, temps_str)):

        mi_dir = f'output_magMI/{gtype}/{gname}/{T_str}'

        subprocess.call(['python3', 'run_greedy_info_maximization.py', \
             str(T_val), \
             f'output_heuristicInfoMax_halfIV/{gtype}/{gname}/{T_str}', \
             #f'output_systemEntropyGreedy/{gtype}/{graph}/T={T:.2f}/heuristic_nodesExcluded_newnodes_greedy', \
             #f'output_systemEntropyGreedy/{gtype}/{graph}/T={T:.2f}/heuristic_nodesExcluded_test', \
             f'networkData/{gtype}/{gname}/{gname}.gpickle', \
             '--k_max', str(int(N/2)), \
             #'--k_max', '8', \
             #'--k_min', '7', \
             '--trials', '1', \
             '--snapshots', '100000', \
             '--excludeNodes', \
             '--heuristic', mi_dir, \
             '--magSide', magSides[j]])
    #"""
