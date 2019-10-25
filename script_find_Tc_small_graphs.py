#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

import subprocess, os

for N in [30, 40, 50, 60, 70]:
    k=N*0.05*1.2
    gtype=f'small_graphs/N={N}_p=0.05'

    for i in range(10):
        gname=f'ER_k={k:.2f}_N={N}_v{i}'
        if not os.path.exists(f'tempsData/{gtype}/{gname}/{gname}_tempsResults.pickle'):
            subprocess.call(['python3', 'find_Tc.py', \
                f'tempsData/{gtype}/{gname}', \
                f'networkData/{gtype}/{gname}/{gname}.gpickle'])
