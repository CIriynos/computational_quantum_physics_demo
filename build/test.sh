#!/bin/bash
gprof ./CQP_DEMO | ../script/gprof2dot.py -w -s | dot -Tpng -o test_output.png    