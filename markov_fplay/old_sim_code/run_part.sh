#!/bin/bash

for i in {0..3000..5}
do
    #echo "emplum_$i"
    python3 playersN_partial_info.py $i &
done
