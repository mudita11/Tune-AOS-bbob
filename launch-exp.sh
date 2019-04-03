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
        PROC_ID=$!
        sleep 1
        if kill -0 "$PROC_ID" >/dev/null 2>&1; then
            echo "Executing on $EXECDIR with PID=${PROC_ID} and output in output-${mutation}-${targetrunner}"
        else
            echo "ERROR executing on $EXECDIR see output in output-${mutation}-${targetrunner}"
        fi
    done
done
