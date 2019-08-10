#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

import subprocess, re, os, glob, itertools
import numpy as np
import networkx as nx
from Utils import IO

N=50
gtype = f'small_graphs/N={N}_p=0.05'
vs = list(range(1,10))

for v in vs:
    graph = f'ER_k={0.05*1.2*N:.2f}_N={N}_v{v}'

    T_result = IO.TcResult.loadFromPickle(f'DataTc_new/{gtype}/{graph}', f'{graph}_Tc_results.pickle')

    #Ts = [T_result.T_low, T_result.T_c, T_result.T_high]
    #magSides = ['pos', 'fair', 'fair']

    #Ts = [T_result.T_low, T_result.T_c]
    #magSides = ['pos', 'fair']

    Ts = [T_result.T_c]
    magSides = ['fair']

    for j, T in enumerate(Ts):
        for i in range(10):
            subprocess.call(['python3', 'run_covering.py', \
                 f'{T:.2f}', \
                 #f'output_systemEntropyGreedy/{gtype}/{graph}/T={T:.2f}_nodesNotExcluded/10_trials/test', \
                 f'output_systemEntropyGreedy/{gtype}/{graph}/T={T:.2f}/10_trials_nodesExcluded', \
                 f'networkData/{gtype}/{graph}/{graph}.gpickle', \
                 '--k_max', str(int(N/2)), \
                 '--trials', '1', \
                 '--snapshots', '100000', \
                 '--excludeNodes', \
                 #'--onlyRandom', \
                 '--magSide', magSides[j]])
