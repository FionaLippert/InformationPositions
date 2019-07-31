#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

import subprocess, re, os, glob, itertools
import numpy as np
import networkx as nx
from Utils import IO

N=30
gtype = f'small_graphs/N={N}_p=0.05'
graph = f'ER_k={0.05*1.2*N:.2f}_N={N}_v7'

T_result = IO.TcResult.loadFromPickle(f'DataTc_new/{gtype}/{graph}', f'{graph}_Tc_results.pickle')

Ts = [T_result.T_low, T_result.T_c, T_result.T_high]
magSides = ['pos', 'fair', 'fair']

Ts = [T_result.T_c]#, T_result.T_high]
magSides = ['fair']#, 'fair']

for j, T in enumerate(Ts):
    for i in range(1):
        subprocess.call(['python3', 'run_covering.py', \
             f'{T:.2f}', \
             f'output_systemEntropyGreedy/{gtype}/{graph}/T={T:.2f}_nodesNotExcluded/10_trials', \
             #f'output_systemEntropyGreedy/{gtype}/{graph}/T={T:.2f}/10_trials', \
             f'networkData/{gtype}/{graph}/{graph}.gpickle', \
             '--k_max', str(N), \
             '--trials', '1', \
             '--snapshots', '100000', \
             #'--excludeNodes', \
             #'--onlyRandom', \
             '--magSide', magSides[j]])
