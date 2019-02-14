#!/usr/bin/env python3
###############################################################################
# This script is the command that is executed every run.
# Check the examples in examples/
#
# This script is run in the execution directory (execDir, --exec-dir).
#
# PARAMETERS:
# argv[1] is the candidate configuration number
# argv[2] is the instance ID
# argv[3] is the seed
# argv[4] is the instance name
# The rest (argv[5:]) are parameters to the run
#
# RETURN VALUE:
# This script should print one numerical value: the cost that must be minimized.
# Exit with 0 if no error, with 1 in case of error
###############################################################################

import datetime
import os.path
import os
import subprocess
import sys

exe = "python3 ../bin/DE_AOS.py bbob 10000 1 1"

fixed_params = " "

if len(sys.argv) < 5:
    print ("\nUsage: ./target-runner.py <candidate_id> <instance_id> <seed> <instance_path_name> <list of parameters>\n")
    sys.exit(1)

def target_runner_error(msg):
    now = datetime.datetime.now()
    print(str(now) + " error: " + msg)
    sys.exit(1)

# Get the parameters as command line arguments.
candidate_id = sys.argv[1]
instance_id = sys.argv[2]
seed = sys.argv[3]
instance = sys.argv[4]; #print("inst1",instance)
cand_params = sys.argv[5:]

# c1=cand_params[1]
# c2=cand_params[3]
# c3=cand_params[5]
# # print(c1, c2, c3, c4, c5, c6)

# cand_params=[str(c1), str(c2), str(c3), instance]

# Define the stdout and stderr files.
out_file = "c" + str(candidate_id) + "-" + str(instance_id) + ".stdout"
err_file = "c" + str(candidate_id) + "-" + str(instance_id) + ".stderr"

# Build the command, run it and save the output to a file,
# to parse the result from it.
# 
# Stdout and stderr files have to be opened before the call().
#
# Exit with error if something went wrong in the execution.

command = " ".join([exe] + cand_params + [instance])
print(command)

outf = open(out_file, "w")
errf = open(err_file, "w")
return_code = subprocess.call(command, shell=True,stdout = outf, stderr = errf)
outf.close()
errf.close()

if return_code != 0:
    now = datetime.datetime.now()
    print(str(now) + " error: command returned code " + str(return_code))
    sys.exit(1)

if not os.path.isfile(out_file):
    now = datetime.datetime.now()
    print(str(now) + " error: output file "+ out_file  +" not found.")
    sys.exit(1)

# get file
filename = 'bbobexp_f1_DIM20_i1-run1.dat'

import numpy as np

points = np.loadtxt(filename, comments="%", usecols=(0,2))

# See README.txt to install this
from pygmo import hypervolume

# max fe_evals * 10,
# TODO: normalize points to [0, 1], then use [1.1, 1.1] as ref
ref_point = [12608 * 10, 8.977281728e+01 * 10]
hv = hypervolume(points)
cost = hv.compute(ref_point)
#cost=[line.rstrip('\n') for line in open(out_file)][-8]

# This is an example of reading a number from the output.
# It assumes that the objective value is the first number in
# the first column of the last line of the output.
# from http://stackoverflow.com/questions/4703390

# print("Cost:= ",cost)
print(cost)

sys.exit(0)
#print("End of target-runner")
