#!/bin/bash

for i in `seq 0 9`;
do
        mkdir 'ER_k=3.60_N=60_v'$i
        mv 'ER_k=3.60_N=60_v'$i'_Tc_results.pickle' 'ER_k=3.60_N=60_v'$i
        mv 'ER_k=3.60_N=60_v'$i'_Tc.txt' 'ER_k=3.60_N=60_v'$i
done
