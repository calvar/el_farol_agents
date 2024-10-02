#!/bin/bash

for i in {0..1000..5}
do
    #echo "emplum_$i"
    python3 playersN.py $i
done
