#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

import subprocess, re, os, glob, itertools
import numpy as np
from Utils import IO

gtype   = '2D_grid' #'ER'
gname   = '2D_grid_L=32' #'ER_k=2.0_N=1000'
version = 0


#gpath=f'networkData/{gtype}/{gname}/{gname}_v{version}.gpickle'
gpath=f'networkData/{gtype}/{gname}.gpickle'

#Tc_path = f'DataTc_new/{gtype}/{gname}/{gname}_v{version}_Tc.txt'
Tc_path = f'DataTc_new/{gtype}/{gname}_Tc.txt'
with open(Tc_path) as f:
   for cnt, line in enumerate(f):
       T = re.findall(r'[\d\.\d]+', line)[0]
       print(f'run T={T}')
       if cnt == 0:
           magSide = 'pos'
       else:
           magSide = 'fair'

       if cnt == 2:
           print('2.57')
           nodes = f'networkData/{gtype}/{gname}_sample_nodes_weighted_10.npy'
           subprocess.call(['python3', 'run_condMI_nodelist.py', \
                    #str(T), \
                    '2.57', \
                    f'output_final/{gtype}/{gname}/vector/T={T}', \
                    gpath, \
                    #'528', \
                    '--nodes', nodes, \
                    '--maxDist', '2' , \
                    '--maxCorrTime', '100', \
                    '--minCorrTime', '100', \
                    '--snapshots', '100', \
                    '--magSide', magSide #, \
                    #'--runs', '10'
                    ])
