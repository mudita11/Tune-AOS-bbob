#!/bin/bash
set -e

# budget=100000
budget=100
seed=42
aos_exe="python3 ./bin/DE_AOS.py bbob $budget 1 1"
de_defaults="--seed $seed --top_NP 0.05 --FF 0.5 --CR 1.0 --NP 200 --train_test test"
MUTATIONS="DE/rand/1 DE/rand/2 DE/rand-to-best/2 DE/current-to-rand/1 DE/current_to_pbest DE/current_to_pbest_archived DE/best/1 DE/current_to_best/1 DE/best/2"

launch() {
    echo $@
    $@
}

rm -rf exdata/

#algos="COBRA ADOPP ADOPP_ext Hybrid Op_adapt APOS PDP Adapt_NN MA_S2 Dyn_GEPv1 Dyn_GEPv2 DMAB ExDMAB ExPM ExAP"
algos="COBRA ADOPP ADOPP_ext Hybrid Op_adapt APOS PDP Adapt_NN MA_S2 Dyn_GEPv1 Dyn_GEPv2 DMAB ExDMAB ExPM ExAP Compass"
for algo in $algos; do
    algo_params="--mutation aos --known-aos $algo"
    launch $aos_exe $de_defaults ${algo_params} --result_folder ${algo//_/-} --name ${algo//_/-}
done

launch $aos_exe $de_defaults --mutation random --result_folder "DE-Random" --name "DE-Random"

for mutation in $MUTATIONS; do
    launch $aos_exe $de_defaults --mutation $mutation --result_folder ${mutation//\//-} --name $mutation
done

PPFOLDER="ppdata"

# Coco post-processing.
cocopp() {
    NAME="$1"
    #rm -rf $PPFOLDER
    shift 1
    echo "running: python3 -m cocopp -o ${PPFOLDER}/${NAME} $@"
    python3 -m cocopp -o ${PPFOLDER}/${NAME} --no-browser $@
}

rm -rf ppdata
cocopp mutations $(echo exdata/*bbob*budget${budget}xD/DE-[crb]*)
EXDATA="$(echo exdata/*bbob*budget${budget}xD/DE-Random)"
for algo in $algos; do
    EXDATA="$EXDATA $(echo exdata/*bbob*budget${budget}xD/${algo//_/-})"
done
cocopp algos $EXDATA



DIRS=$(echo ${PPFOLDER}/*/)
root=$(pwd)
COPY="cp -f"

for indir in $DIRS; do
    NAME=${indir//${PPFOLDER}\//}
    NAME=${NAME//\//}
    echo "Processing $indir"
    outdir="${root}/coco/$NAME"
    pushd $indir
    insubdir=$(echo */)
    echo $insubdir
    mkdir -p ${outdir}/${insubdir}
    $COPY cocopp_commands.tex ${outdir}/
    $COPY ${insubdir}/pptables_f*D.tex ${outdir}/${insubdir}/
    $COPY ${insubdir}/pprldmany_*D_*.tex ${outdir}/${insubdir}/
    $COPY ${insubdir}/pprldmany_*D_*.pdf ${outdir}/${insubdir}/
    popd
    echo "Tables and figures written to $outdir"
done
