#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

import subprocess, re, os, glob, itertools
import numpy as np
from Utils import IO

N = 50
gtype=f'small_graphs/N={N}_p=0.05'


for v in range(10):
    graph=f'ER_k={0.05*1.2*N:.2f}_N={N}_v{v}'

    results = IO.TcResult.loadFromPickle(f'DataTc_new/{gtype}/{graph}', f'{graph}_Tc_results.pickle')

    for i, T in enumerate([results.T_low, results.T_c, results.T_high]):
           print(f'run T={T:.2f}')
           if i == 0:
               magSide = 'pos'
           else:
               magSide = 'fair'

           if i == 2:

               print('start with 10 runs')

               subprocess.call(['python3', 'run_condMI_nodelist.py', \
                        str(T), \
                        f'output_final/{gtype}/{graph}/vector_snapshots=100_runs=10/T={T:.2f}', \
                        f'networkData/{gtype}/{graph}/{graph}.gpickle', \
                        '--nodes', f'networkData/{gtype}/{graph}/{graph}_nodes.npy', \
                        '--maxCorrTime', '100', \
                        '--minCorrTime', '100', \
                        '--snapshots', '100', \
                        '--threshold', '0.001', \
                        '--runs', '10', \
                        '--magSide', magSide])
