#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

import subprocess, re, os, glob, itertools
import numpy as np
from Utils import IO

gtype   = 'small_graphs/N=70_p=0.05'
gname   = 'ER_k=4.20_N=70'


for i in range(10):

    gpath   = f'{gtype}/{gname}_v{i}'
    graph=f'networkData/{gpath}/{gname}_v{i}.gpickle'
    results = IO.TcResult.loadFromPickle(f'DataTc_new/{gtype}/{gname}', f'{gname}_Tc_results.pickle')

    for i, T in enumerate([results.T_o, results.T_c, results.T_d]):
           print(f'run T={T:.2f}')
           if i == 0:
               magSide = 'pos'
           else:
               magSide = 'fair'

           subprocess.call(['python3', 'run_magMI_estimation.py', \
                    str(T), \
                    f'output/{gpath}/magMI/T={T}', \
                    graph, \
                    '--neighboursDir', f'networkData/{gpath}', \
                    '--magSide', magSide ])
