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


results = IO.TempsResult.loadFromPickle(f'tempsData/{gtype}/{gname}', f'{gname}_tempsResults.pickle')

for i, (T_val, T_str) in enumerate(zip([results.T_o, results.T_c, results.T_d], ['T_o', 'T_c', 'T_d'])):
       print(f'run T={T_str}')
       if i == 0:
           magSide = 'pos'
       else:
           magSide = 'fair'

       subprocess.call(['python3', 'run_magMI_estimation.py', \
                str(T_val), \
                f'output_magMI/{gtype}/{gname}/{T_str}', \
                graph, \
                '--neighboursDir', f'networkData/{gpath}', \
                '--magSide', magSide ])
