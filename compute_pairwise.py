#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'


from Models import fastIsing
from Toolbox import infcy
from Utils import IO
import networkx as nx, itertools, scipy, time, subprocess, \
                os, pickle, sys, argparse, multiprocessing as mp
import itertools
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
from scipy import stats

nthreads = mp.cpu_count() - 1

parser = argparse.ArgumentParser(description='compute pairwise correlation and MI based on system snapshots from MC chain')
parser.add_argument('path', type=str, help='path to full system snapshots')

if __name__ == '__main__':

    args = parser.parse_args()
    fullSnapshots = np.load(args.path)

    MI, corr = infcy.runMI(model, nodes, fullSnapshots.reshape((args.repeats*args.numSamples, -1)), distMax=maxDist)
    np.save(os.path.join(targetDirectory, f'MI_pairwise_nodes_{now}.npy'), MI)
    np.save(os.path.join(targetDirectory, f'corr_pairwise_nodes_{now}.npy'), corr)
