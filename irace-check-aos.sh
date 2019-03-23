#!/bin/bash
set -e
set -o pipefail

variants="AdapSS.txt  ADOPP_ext.txt  AP.txt  compass.txt  FAUC.txt  hybrid_GA.txt  PDP_GP.txt  Random.txt  rec_pm.txt  scenario.txt  SR.txt "
for k in $variants; do
    irace --check --parameter-file $k
done
