#!/bin/bash

variants="AdapSS.txt  ADOPP_ext.txt  AP.txt  compass.txt  FAUC.txt  hybrid_GA.txt  PDP_GP.txt  Random.txt  rec_pm.txt  SR.txt"
targetrunners="best target-vs-fe"
for variant in $variants; do
    for targetrunner in $targetrunners; do
        EXECDIR=execdir-$variant-$targetrunner
        mkdir $EXECDIR
        irace --parameter-file ${variant}.txt --exec-dir $EXECDIR --target-runner target-runner-${targetrunner}.py &> output-${variant}-${targetrunner} &
    done
done
