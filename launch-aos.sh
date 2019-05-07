#!/bin/bash

fes=10^5
file="configurations-aos.txt"
de="./bin/DE_AOS.py bbob $fes 1 1 --train_test test"

cat $file | parallel --verbose --eta -j 1 --colsep '' "$de {} --result_folder de-run-{#} > aos-run-{#}.output"
