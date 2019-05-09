#!/bin/bash

array=( 100 150 200 250 300 350 400 450 500 )
for i in "${array[@]}"
do
  python3 LISA_run_jointMI_nodelist.py 1.46 output_uncertainty/ER/ER_k\=2.0_N\=1000_v0/avgMI_numBins/$i networkData/ER/ER_k\=2.0_N\=1000_v0.gpickle networkData/ER/ER_k\=2.0_N\=1000_v0_nodes.npy --neighboursDir output_uncertainty/ER/ER_k\=2.0_N\=1000_v0/ --numSamples 10000 --runs 1 --bins $i
done
