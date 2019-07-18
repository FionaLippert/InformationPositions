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

Tc_path = f'DataTc_new/{gtype}/{gname}/{gname}_v{version}_Tc.txt'
with open(Tc_path) as f:
   for cnt, line in enumerate(f):
       T = re.findall(r'[\d\.\d]+', line)[0]
       print(f'run T={T}')
       if cnt == 0:
           magSide = 'pos'
       else:
           magSide = 'fair'

       if cnt > 1:


           subprocess.call(['python3', 'run_jointMI_nodelist.py', \
                    str(T), \
                    f'output_final/{gtype}/{gname}/{gname}_v{version}/magMI/T={T}', \
                    gpath, \
                    '--neighboursDir', f'networkData/{gtype}/{gname}/{gname}_v{version}', \
                    '--magSide', magSide ])
