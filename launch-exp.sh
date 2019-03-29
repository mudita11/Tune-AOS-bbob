#!/bin/bash

#mutations="DE_rand_1 DE_rand_2 DE_rand_to_best_2 DE_current_to_rand_1"
mutations="DE_rand_1"
#targetrunners="best best-vs-fe target-vs-fe"
targetrunners="best-vs-fe"
for mutation in $mutations; do
    for targetrunner in $targetrunners; do
        EXECDIR=execdir-$mutation-$targetrunner
        mkdir $EXECDIR
        irace --parameter-file ${mutation}.txt --exec-dir $EXECDIR --target-runner target-runner-${targetrunner}.py --parallel 3 &> output-${mutation}-${targetrunner} &
    done
done
