#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

import subprocess, re, os, glob, itertools
import numpy as np
from Utils import IO

gtype   = 'ER/ER_k=2.0_N=1000'
version = 0


gpath=f'networkData/{gtype}/{gname}/{gname}.gpickle'

results = IO.TempsResult.loadFromPickle(f'tempsData/{gtype}/{gname}', f'{gname}_tempsResults.pickle')

for i, (T_val, T_str) in enumerate(zip([results.T_o, results.T_c, results.T_d], ['T_o', 'T_c', 'T_d'])):
       print(f'run T={T_str}')
       if i == 0:
           magSide = 'pos'
       else:
           magSide = 'fair'
       nodes = f'networkData/{gtype}/{gname}/{gname}_sample_nodes_weighted_10.npy'

       subprocess.call(['python3', 'run_condMI_nodelist.py', \
                str(T_val), \
                f'output_snapshotMI/{gtype}/{gname}/{T_str}', \
                gpath, \
                '--nodes', nodes, \
                '--threshold', '0.001', \
                '--maxCorrTime', '100', \
                '--minCorrTime', '100', \
                '--magSide', magSide])
