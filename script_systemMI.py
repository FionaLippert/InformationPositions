#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

import subprocess, os, glob, itertools
import numpy as np
import networkx as nx
from Utils import IO


N=30
k=N*0.05*1.2
gtype=f'small_graphs/N={N}_p=0.05'


for i in range(10):

    gname=f'ER_k={k:.2f}_N={N}_v{i}'

    G = nx.read_gpickle(f'networkData/{gtype}/{gname}/{gname}.gpickle')
    np.save(f'networkData/{gtype}/{gname}/{gname}_nodes.npy', list(G))

    results = IO.TcResult.loadFromPickle(f'tempsData/{gtype}/{gname}', f'{gname}_tempsResults.pickle')

    for i, T in enumerate([results.T_o, results.T_c, results.T_d]):
           print(f'run T={T:.2f}')
           if i == 0:
               magSide = 'pos'
           else:
               magSide = 'fair'

           subprocess.call(['python3', 'run_systemMI_estimation.py', \
                str(T), \
                f'output_systemEntropy/{gtype}/{gname}/T={T:.2f}', \
                f'networkData/{gtype}/{gname}/{gname}.gpickle', \
                '--nodes', f'networkData/{gtype}/{gname}/{gname}_nodes.npy', \
                '--k', '1', \
                '--trials', '1', \
                '--snapshots', '100000', \
                '--excludeNodes', \
                '--magSide', magSide])
