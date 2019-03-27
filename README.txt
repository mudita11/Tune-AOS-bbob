# Install cocoex:

# Follow these steps taken from https://numbbo.github.io/coco-doc/ :

cd ..
git clone https://github.com/numbbo/coco.git # get coco using git
cd coco
python3 do.py run-python install-user # install Python experimental module cocoex 
python3 do.py install-postprocessing install-user # install post-processing 


# Install pygmo

pip3 install --user pygmo

# Run:

irace --check


target-runner-target-vs-fe.py is same as target-runner-hv that calculates the area under the curve generated using trace file. The ECDF graph represents log10(FEvals/dim) vs fraction of targets solved for a problem. 

target-runner-best-vs-fe.py calculates the area under the curve generated using trace file. The ECDF graph represents log10(FEvals/dim) vs best fitness seen for different targets for a problem. 

target-runner-best.py receives the best fitness value seen for a problem. 