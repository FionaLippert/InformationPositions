#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'


from Models import fastIsing
from Toolbox import simulation
from Utils import IO
import networkx as nx, itertools, scipy,\
        os, pickle, h5py, sys, multiprocessing as mp, json,\
        datetime, sys
import time
import timeit
from matplotlib.pyplot import *
from numpy import *
from tqdm import tqdm
from scipy import optimize, ndimage
import glob
import argparse

path = 'DataTc_new/BA/BA_m=3_N=1000/BA_m=3_N=1000_v0_Tc.txt'
with open(path, 'a') as f:
    f.write('\n')
