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

N=30
gtype = f'small_graphs/N={N}_p=0.05'
#vs = [1] #list(range(6,10))
vs = list(range(10))

for v in vs:
    graph = f'ER_k={0.05*1.2*N:.2f}_N={N}_v{v}'

    T_result = IO.TcResult.loadFromPickle(f'DataTc_new/{gtype}/{graph}', f'{graph}_Tc_results.pickle')

    #Ts = [T_result.T_low, T_result.T_c, T_result.T_high]
    #magSides = ['pos', 'fair', 'fair']

    #Ts = [T_result.T_low, T_result.T_c]
    #magSides = ['pos', 'fair']

    Ts = [T_result.T_low]
    magSides = ['pos']

    """
    for j, T in enumerate(Ts):

        for k in range(1, 4):
            subprocess.call(['python3', 'run_covering.py', \
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
            subprocess.call(['python3', 'run_covering.py', \
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
    for j, T in enumerate(Ts):

        mi_dir = f'output_final/{gtype}/{graph}/magMI/T={T:.2f}'

        subprocess.call(['python3', 'run_covering.py', \
             f'{T:.2f}', \
             f'output_systemEntropyGreedy/{gtype}/{graph}/T={T:.2f}/heuristic_nodesExcluded_iv_greedy', \
             #f'output_systemEntropyGreedy/{gtype}/{graph}/T={T:.2f}/heuristic_nodesExcluded_newnodes_greedy', \
             #f'output_systemEntropyGreedy/{gtype}/{graph}/T={T:.2f}/heuristic_nodesExcluded_test', \
             f'networkData/{gtype}/{graph}/{graph}.gpickle', \
             '--k_max', str(int(N/2)), \
             #'--k_max', '8', \
             #'--k_min', '7', \
             '--trials', '1', \
             '--snapshots', '100000', \
             '--excludeNodes', \
             '--heuristic', mi_dir, \
             '--magSide', magSides[j]])
    #"""
