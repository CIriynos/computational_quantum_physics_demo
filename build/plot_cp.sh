#!/bin/bash
cd ..
python3 ./script/plot_cp.py "$1" "$2"
code ./figure/"$1_cp.png"
cd ./build/