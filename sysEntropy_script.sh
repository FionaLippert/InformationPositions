#!/bin/bash
T=1.48
#snapshots=( 10 100 1000 10000 )
snapshots=( 100000 )

graph="small_graphs/N=50/ER_k=2.80_N=50"
for i in "${snapshots[@]}"
do
  python3 LISA_run_systemEntropy.py $T output_systemEntropy/$graph/T\=$T/system/$i networkData/$graph.gpickle --snapshots $i --repeats 10
done
