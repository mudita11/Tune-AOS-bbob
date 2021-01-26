#!/bin/bash
PPFOLDER="ppdata"
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

