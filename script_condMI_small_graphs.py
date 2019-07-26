#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

import subprocess, re, os, glob, itertools
import numpy as np
from Utils import IO

gtype='small_graphs/N=50'

#k_list = [2.80, 2.88, 2.16, 2.72, 3.16, 3.24, 3.52, 3.64, 1.96, 2.28]
k_list = [3.16, 3.24, 3.52, 3.64, 1.96, 2.28]
#k_list = [2.72, 3.16, 3.24, 3.52, 3.64]
k_list = [3.16]

for k in k_list:
    graph=f'ER_k={k:.2f}_N=50'

    Tc_path = f'DataTc_new/{gtype}/{graph}_Tc.txt'
    with open(Tc_path) as f:
       for cnt, line in enumerate(f):
           T = re.findall(r'[\d\.\d]+', line)[0]
           print(f'run T={T}')
           if cnt == 0:
               magSide = 'pos'
           else:
               magSide = 'fair'

           if cnt == 2:
               subprocess.call(['python3', 'run_condMI_nodelist.py', \
                        str(T), \
                        f'output_final/{gtype}/{graph}/vector/T={T}', \
                        f'networkData/{gtype}/{graph}.gpickle', \
                        '--nodes', f'networkData/{gtype}/{graph}_nodes.npy', \
                        '--maxCorrTime', '100', \
                        '--minCorrTime', '100', \
                        '--magSide', magSide])
