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


results = IO.TcResult.loadFromPickle(f'DataTc_new/{gtype}/{gname}', f'{gname}_Tc_results.pickle')

for i, T in enumerate([results.T_o, results.T_c, results.T_d]):
       print(f'run T={T:.2f}')
       if i == 0:
           magSide = 'pos'
       else:
           magSide = 'fair'

       subprocess.call(['python3', 'run_magMI_estimation.py', \
                str(T), \
                f'output/{gpath}/magMI/T={T:.2f}', \
                graph, \
                '--neighboursDir', f'networkData/{gpath}', \
                '--magSide', magSide ])
