#!/bin/bash
cd ..
python3 ./script/plot.py "$1"
code ./figure/"$1.png"
cd ./build/