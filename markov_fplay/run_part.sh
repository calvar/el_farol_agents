#!/bin/bash

for i in {3005..4500..5}
do
    #echo "emplum_$i"
    python3 playersN_partial_info.py $i &
done
