#!/bin/bash

#mutations="DE_rand_1 DE_rand_2"
mutations="DE_rand_2"
targetrunners="best target-vs-fe"
for mutation in $mutations; do
    for targetrunner in $targetrunners; do
        EXECDIR=execdir-$mutation-$targetrunner
        mkdir $EXECDIR
        irace --parameter-file ${mutation}.txt --exec-dir $EXECDIR --target-runner target-runner-${targetrunner}.py &> output-${mutation}-${targetrunner} &
    done
done
