#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'


from Models import fastIsing
from Toolbox import infcy
from Utils import IO
import networkx as nx, itertools, scipy,\
        os, pickle, h5py, sys, multiprocessing as mp, json,\
        datetime, sys, argparse
import time
import timeit
from matplotlib.pyplot import *
from numpy import *
from tqdm import tqdm
from scipy import optimize, ndimage
import glob
close('all')

parser = argparse.ArgumentParser(description='determine mixing and correlation time')
parser.add_argument('T_min', type=float, help='minimum temperature')
parser.add_argument('T_max', type=float, help='maximum temperature')
parser.add_argument('T_num', type=float, help='number of different temperatures in between')
parser.add_argument('dir', type=str, help='target directory')
parser.add_argument('graph', type=str, help='target directory')



def find_Tc_gaussian(sus, temps, sigma=5):
    return temps[np.argmax(ndimage.filters.gaussian_filter1d(sus, sigma))]


if __name__ == '__main__':

    args = parser.parse_args()

    targetDirectory = args.dir
    os.makedirs(targetDirectory, exist_ok=True)


    ensemble = [g for g in glob.iglob(f'networkData/{args.graph}*.gpickle')]
    print(ensemble)

    all_Tc = np.zeros(len(ensemble))


    temps = linspace(args.T_min, args.T_max, args.T_num)

    nSamples      = int(1e4) #int(1e6)
    burninSamples = int(1e4) # int(1e6)
    magSide       = ''
    updateType    = 'async'

    os.makedirs(targetDirectory, exist_ok=True)

    settings = dict(
        nSamples         = nSamples, \
        burninSamples    = burninSamples, \
        updateMethod     = updateType, \
        magSide          = magSide
        )
    IO.saveSettings(targetDirectory, settings)


    for i, g in enumerate(ensemble):

        graph = nx.read_gpickle(g)
        filename = os.path.split(g)[-1].strip('.gpickle')

        modelSettings = dict(\
                             graph       = graph,\
                             updateType  = updateType,\
                             magSide     = magSide
                             )
        model = fastIsing.Ising(**modelSettings)

        states = infcy.simulateGetStates(model, \
                        burninSteps = burninSamples, \
                        nSamples = nSamples)

        p_i = (np.mean(states, axis=0) + 1) / 2 # spin probs
        HX = -np.sum(p_i * np.log2(p_i)) # spin entropies


        tmp = dict( \
                temps = temps, \
                magnetization = mag, \
                absMagnetization = mag_abs, \
                susceptibility = sus, \
                binder = binder, \
                spinEntropies = HX)
        IO.savePickle(targetDirectory, f'{filename}_results', tmp)
