#!/bin/bash

for i in {0..3000..5}
do
    #echo "emplum_$i"
    python3 playersN_red_mat.py $i &
done
