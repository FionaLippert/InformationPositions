#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

from Toolbox import infoTheory
import argparse
import numpy as np

nthreads = mp.cpu_count() - 1

parser = argparse.ArgumentParser(description='compute pairwise correlation and MI based on system snapshots from MCMC simulation')
parser.add_argument('path', type=str, help='path to full system snapshots')

if __name__ == '__main__':

    args = parser.parse_args()
    fullSnapshots = np.load(args.path)

    MI, corr = infoTheory.pairwiseMI_allNodes(model, nodes, fullSnapshots.reshape((args.repeats*args.numSamples, -1)), distMax=maxDist)
    np.save(os.path.join(targetDirectory, f'MI_pairwise_nodes_{now}.npy'), MI)
    np.save(os.path.join(targetDirectory, f'corr_pairwise_nodes_{now}.npy'), corr)
