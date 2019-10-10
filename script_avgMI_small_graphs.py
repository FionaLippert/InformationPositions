#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

import subprocess, re, os, glob, itertools
import numpy as np
from Utils import IO

gtype   = 'small_graphs/N=70_p=0.05'
gname   = 'ER_k=4.20_N=70'


for i in range(6,10):

    gpath   = f'{gtype}/{gname}_v{i}'
    graph=f'networkData/{gpath}/{gname}_v{i}.gpickle'
    Tc_path = f'DataTc_new/{gpath}/{gname}_v{i}_Tc.txt'

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
                    f'output_final/{gpath}/avg/T={T}', \
                    graph, \
                    '--neighboursDir', f'networkData/{gpath}', \
                    '--magSide', magSide ])