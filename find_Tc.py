#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'


from Models import fastIsing
from Toolbox import simulation
from Utils import IO

import networkx as nx
import multiprocessing as mp
import itertools, scipy, os, warnings, \
        pickle, h5py, sys, json, sys
import numpy as np
from tqdm import tqdm
from scipy import optimize, ndimage
import glob
import argparse

parser = argparse.ArgumentParser(description='find the critical temperature T_c \
            and temperatures T_o and T_d in the orgered and disordered phease \
            of the Ising Model on a given network structure')
parser.add_argument('dir', type=str, help='target directory')
parser.add_argument('graph', type=str, help='path to graph or ensemble of graphs')
parser.add_argument('--minT', type=float, default=0.1,
            help='minimum temperature for simulations')
parser.add_argument('--maxT', type=float, default=5.0,
            help='maximum temperature for simulations')
parser.add_argument('--numT', type=int, default=1000,
            help='number of different temperature values')


def find_Tc_gaussian(sus, temps, sigma=5):
    sus_smoothed = ndimage.filters.gaussian_filter1d(sus, sigma)
    Tc_idx = np.argmax(sus_smoothed)
    Tc = temps[Tc_idx]
    return Tc, Tc_idx

# sigmoid function for magnetization fitting
def sig(x, b, d, c):
    return 1 / (1 + np.exp(b*(x-d))) + c


if __name__ == '__main__':

    args = parser.parse_args()

    if args.graph.endswith('.gpickle'):
        ensemble = [args.graph]
    else:
        ensemble = [g for g in glob.iglob(f'{args.graph}*.gpickle')]
    print(f'the following graphs have been found: {ensemble}')


    temps = np.linspace(args.minT, args.maxT, args.numT)
    nSamples      = int(1e4)
    burninSteps   = int(1e4)
    magSide       = ''
    updateType    = 'async'


    targetDirectory = f'{os.getcwd()}/{args.dir}'
    os.makedirs(targetDirectory, exist_ok=True)

    settings = dict(
        nSamples         = nSamples, \
        burninSteps      = burninSteps, \
        updateMethod     = updateType
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


        mags, sus, binder, abs_mags = simulation.magnetizationParallel(model, \
                            temps        = temps,        \
                            n            = nSamples,     \
                            burninSteps  = burninSteps)

        Tc, Tc_idx = find_Tc_gaussian(sus, temps)
        a, b = optimize.curve_fit(sig, temps, abs_mags)
        mag_Tc = sig(Tc, *a)

        if np.all(sus < 1e-2):
            warnings.warn(f'Susceptibility values are too small to detect phase \
                            transition. Instead, T_c is estimated based on \
                            the difference between absolute average magnetization \
                            and average absolute magnetization.', Warning)
            symmetry_breaking_idx = np.where( \
                    np.abs(ndimage.filters.gaussian_filter1d(np.abs(mags), 10) \
                    - ndimage.filters.gaussian_filter1d(np.abs(abs_mags), 10)) \
                    > 0.01)[0][0]
            mag_Tc = sig(temps[symmetry_breaking_idx], *a)

        #To = temps[np.where(sig(temps, *a) < (1 + sig(temps[symmetry_breaking_idx], *a)) * 0.5)[0][0]]
        To = temps[np.where(sig(temps, *a) < (1 + mag_Tc) * 0.5)[0][0]]

        try:
            Td = temps[np.where(sig(temps, *a) < mag_Tc * 0.5)[0][0]]
        except:
            warnings.warn(f'T_d could not be estimated from sigmoid fit. \
                            Raw magnetization data is used instead', Warning)
            Td = temps[np.where(abs_mags < mag_Tc * 0.5)[0][0]]

        print(f'T_o (ordered phase)    = {To:.2f}')
        print(f'T_c (critical)         = {Tc:.2f}')
        print(f'T_d (disordered phase) = {Td:.2f}')

        result = IO.TempsResult(temps, mags, abs_mags, sus, binder, Tc, Td, To, filename)
        result.saveToPickle(targetDirectory)
