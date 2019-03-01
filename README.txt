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
