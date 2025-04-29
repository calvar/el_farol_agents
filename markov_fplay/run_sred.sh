#!/bin/bash

for i in {2915..3000..5}
do
    #echo "emplum_$i"
    python3 playersN_semred_mat.py $i &
done
