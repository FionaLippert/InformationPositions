#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

import subprocess, re, os, glob, itertools
import numpy as np
from Utils import IO

gtype   = 'ER'
gname   = 'ER_k=2.0_N=1000'
version = 0


gpath=f'networkData/{gtype}/{gname}/{gname}_v{version}.gpickle'

results = IO.TcResult.loadFromPickle(f'tempsData/{gtype}/{gname}', f'{gname}_v{version}_tempsResults.pickle')

for i, T in enumerate([results.T_o, results.T_c, results.T_d]):
       print(f'run T={T:.2f}')
       if i == 0:
           magSide = 'pos'
       else:
           magSide = 'fair'
       nodes = f'networkData/{gtype}/{gname}/{gname}_v{version}_sample_nodes_weighted_10.npy'

       subprocess.call(['python3', 'run_condMI_nodelist.py', \
                str(T), \
                f'output/{gtype}/{gname}/{gname}_v{version}/snapshotMI/T={T:.2f}', \
                gpath, \
                '--nodes', nodes, \
                '--threshold', '0.001', \
                '--maxCorrTime', '100', \
                '--minCorrTime', '100', \
                '--magSide', magSide])
