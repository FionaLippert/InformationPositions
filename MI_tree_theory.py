#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Fiona Lippert'

import numpy as np
from tqdm import tqdm
from scipy import sparse, stats
import itertools

def pCond_theory(dist, beta, num_children, node_state, neighbour_states):

    if dist == 1:
        p = np.exp(beta * node_state * np.sum(neighbour_states))/(np.exp(beta * node_state * np.sum(neighbour_states)) + np.exp(-beta * node_state * np.sum(neighbour_states)))
        return p
    else:
        sum = 0
        for states in itertools.product([1,-1], repeat=num_children):
            P_node_given_children = np.exp(beta * node_state * np.sum(states))/(np.exp(beta * node_state * np.sum(states)) + np.exp(-beta * node_state * np.sum(states)))
            prod = 1
            for idx, n in enumerate(states):
                prod *= pCond_theory(dist-1, beta, num_children, n, neighbour_states[num_children*idx:num_children*(idx+1)])
            sum += P_node_given_children * prod
        return sum

def MI_tree_theory(depth, beta, num_children):

    HX = 1 # uniformly distributed
    HXgivenY = 0

    num_neighbour_states = 2**(num_children**depth)

    for states in tqdm(itertools.product([1,-1], repeat=num_children**depth)):
        pX1 = pCond_theory(depth, beta, num_children, 1, states)
        HXgivenY += stats.entropy([pX1, 1-pX1])

    MI = HX - HXgivenY/num_neighbour_states # states are also uniformly distributed
    return MI

if __name__ == '__main__':

    MI = MI_tree_theory(2, 0.5, 2)
    print(f'MI = {MI}')
    #px = pCond_theory(2, 0.5, 2, 1, [1,1,1,1])
    #print(f'px = {px}')
