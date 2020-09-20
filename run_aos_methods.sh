#!/bin/bash
set -e

# budget=100000
budget=1000
seed=42
aos_exe="python3 ./bin/DE_AOS.py bbob $budget 1 1"
de_defaults="--seed $seed --top_NP 0.05 --FF 0.5 --CR 1.0 --NP 200 --train_test test"
MUTATIONS="DE/rand/1 DE/rand/2 DE/rand-to-best/2 DE/current-to-rand/1 DE/current_to_pbest DE/current_to_pbest_archived DE/best/1 DE/current_to_best/1 DE/best/2"

launch() {
    echo $@
    $@
}

rm -rf exdata
algos="ADOPP "
for algo in $algos; do
    algo_params="--mutation aos --known-aos $algo"
    launch $aos_exe $de_defaults ${algo_params} --result_folder $algo --name ${algo//_/-}
done

launch $aos_exe $de_defaults --mutation random --result_folder "DE-Random" --name "DE-Random"

for mutation in $MUTATIONS; do
    launch $aos_exe $de_defaults --mutation $mutation --result_folder ${mutation//\//-} --name $mutation
done

# Coco post-processing.
rm -rf cocopp
EXDATA=$(echo exdata/*bbob*budget${budget}xD)
echo "running: python3 -m cocopp -o cocopp $EXDATA"
python3 -m cocopp -o cocopp $EXDATA
