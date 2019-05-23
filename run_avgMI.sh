#!/bin/bash

#array=( 100 150 200 250 300 350 400 450 500 )
array=( 8 9)
#for i in "${array[@]}"
#do
#  T=$(cat "DataTc/BA/BA_m=3_N=1000/BA_m=3_N=1000_v$i""_Tc.txt")
#  python3 LISA_run_jointMI_nodelist.py $T output_uncertainty/BA/BA_m\=3_N\=1000_v$i/avgMI_numBins/10000 networkData/BA/BA_m\=3_N\=1000_v$i.gpickle networkData/BA/BA_m\=3_N\=1000_v$i"_nodes.npy" --numSamples 10000 --runs 1 --bins 100
#done

for i in "${array[@]}"
do
  T=$(cat "DataTc/WS/WS_k=4_N=1000/WS_k=4_N=1000_v$i""_Tc.txt")
  python3 LISA_run_jointMI_nodelist.py $T output_uncertainty/WS/WS_k\=4_N\=1000_v$i/avgMI_numBins/10000 networkData/WS/WS_k\=4_N\=1000_v$i.gpickle networkData/WS/WS_k\=4_N\=1000_v$i"_nodes.npy" --numSamples 10000 --runs 1 --bins 100
done
