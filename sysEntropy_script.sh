#!/bin/bash

array=( 2 3 7 8 11 13 19 22 24 26 27 28 29 37 42 44 46 51 53 56 )
for i in "${array[@]}"
do
  python3 LISA_run_systemEntropy.py 0.52 output_systemEntropy/small_graphs/ER_k\=1.90_N\=20/T\=0.52/node_$i networkData/small_graphs/ER_k\=1.90_N\=20.gpickle --snapshots 1000 --repeats 10 --centralNode $i --dist -1
done
