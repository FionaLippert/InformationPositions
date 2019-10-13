#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

import subprocess
import numpy as np
from Utils import IO

N = 50
gtype=f'small_graphs/N={N}_p=0.05'


for v in range(10):
    gname=f'ER_k={0.05*1.2*N:.2f}_N={N}_v{v}'

    results = IO.TcResult.loadFromPickle(f'tempsData/{gtype}/{gname}', f'{gname}_tempsResults.pickle')

    for i, T in enumerate([results.T_o, results.T_c, results.T_d]):
           print(f'run T={T:.2f}')
           if i == 0:
               magSide = 'pos'
           else:
               magSide = 'fair'

           subprocess.call(['python3', 'run_snapshotMI_estimation.py', \
                    str(T), \
                    f'output/{gtype}/{gname}/snapshotMI/T={T:.2f}', \
                    f'networkData/{gtype}/{gname}/{gname}.gpickle', \
                    '--nodes', f'networkData/{gtype}/{gname}/{gname}_nodes.npy', \
                    '--maxCorrTime', '100', \
                    '--minCorrTime', '100', \
                    '--snapshots', '100', \
                    '--threshold', '0.001', \
                    '--magSide', magSide])
