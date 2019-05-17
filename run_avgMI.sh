#!/bin/bash

#array=( 100 150 200 250 300 350 400 450 500 )
array=( 0 1 2 3 4 5 6 7 8 9)
for i in "${array[@]}"
do
  T=$(cat "DataTc/ER/ER_k=4.0_N=1000/ER_k=4.0_N=1000_v$i""_Tc.txt")
  python3 LISA_run_jointMI_nodelist.py $T output_uncertainty/ER/ER_k\=4.0_N\=1000_v$i/avgMI_numBins/10000 networkData/ER/ER_k\=4.0_N\=1000_v$i.gpickle networkData/ER/ER_k\=4.0_N\=1000_v$i"_nodes.npy" --numSamples 10000 --runs 1 --bins 100
done
