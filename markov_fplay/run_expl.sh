#!/bin/bash

for i in {1505..2250..5}
do
    #echo "emplum_$i"
    python3 playersN_exploration.py $i &
done
