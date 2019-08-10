#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

import subprocess, re, os, glob, itertools
import numpy as np
from Utils import IO

gtype   = '2D_grid'
#gtype   = 'ER'
#gtype   = 'crime'
gname   = '2D_grid_L=32'
#gname   = 'WS_k=4_beta=0.2_N=1000'
#gname = 'ER_k=4.0_N=1000'
#gname = 'BA_m=3_N=1000'
#gname = 'unweighted_criminal_after_2012_filtered_weights'

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

       if cnt > 0:
           #T = str(2.50)
           subprocess.call(['python3', 'run_jointMI_nodelist.py', \
                    str(T), \
                    f'output_final/{gpath}/avg_bins=500/T={T}', \
                    graph, \
                    '--neighboursDir', f'networkData/{gpath}', \
                    '--bins', '500', \
                    '--magSide', magSide ])
