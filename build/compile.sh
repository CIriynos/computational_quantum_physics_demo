#!/bin/bash
cmake3 -D CMAKE_C_COMPILER=/usr/bin/gcc -D CMAKE_CXX_COMPILER=/usr/bin/g++ -D CMAKE_BUILD_TYPE=Release ..
cmake3 --build .