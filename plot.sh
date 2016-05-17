#!/usr/bin/env bash

# Based on https://github.com/tambetm/simple_dqn/blob/master/plot.sh 


file=${1%.*}
python src/plot.py --png_file $file.png $*
