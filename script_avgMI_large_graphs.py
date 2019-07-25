#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

import subprocess, re, os, glob, itertools
import numpy as np
from Utils import IO

#gtype   = '2D_grid'
gtype   = 'WS'
#gname   = '2D_grid_L=32'
gname   = 'WS_k=4_beta=0.2_N=1000'

#gpath   = f'{gtype}/{gname}/{gname}_v0'
gpath   = f'{gtype}/{gname}'

graph=f'networkData/{gpath}.gpickle'
Tc_path = f'DataTc_new/{gpath}_Tc.txt'

with open(Tc_path) as f:
   for cnt, line in enumerate(f):
       T = re.findall(r'[\d\.\d]+', line)[0]
       print(f'run T={T}')
       if cnt == 0:
           magSide = 'pos'
       else:
           magSide = 'fair'


       subprocess.call(['python3', 'run_jointMI_nodelist.py', \
                str(T), \
                f'output_final/{gpath}/magMI/T={T}', \
                graph, \
                '--neighboursDir', f'networkData/{gpath}', \
                '--magSide', magSide ])
