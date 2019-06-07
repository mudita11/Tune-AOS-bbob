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

import os.path
import subprocess
import sys
import numpy as np

dim = 20
# fevals * dimension f-evaluations
fevals = 5000 # This the value used by coco for the plots
# Options are: suite_name fevals batch total_batches
exe = "python3 ../bin/DE_AOS.py bbob {} 1 1".format(fevals)

if len(sys.argv) < 5:
    print ("\nUsage: ./target-runner.py <candidate_id> <instance_id> <seed> <instance_path_name> <list of parameters>\n")
    sys.exit(1)

def target_runner_error(msg):
    print(sys.argv[0] + ": error: " + msg)
    sys.exit(1)

# Get the parameters as command line arguments.
candidate_id = sys.argv[1]
instance_id = sys.argv[2]
seed = sys.argv[3]
instance = sys.argv[4]
cand_params = sys.argv[5:]

# Define the stdout and stderr files.
prefix = "c{}-{}-{}".format(candidate_id, instance_id, seed)
out_file = prefix + ".stdout"
err_file = prefix + ".stderr"
trace_file = prefix + "_trace.txt"

# Build the command, run it and save the output to a file,
# to parse the result from it.
# 
# Stdout and stderr files have to be opened before the call().
#
# Exit with error if something went wrong in the execution.

outf = open(out_file, "w")
errf = open(err_file, "w")
command = " ".join([exe, "-i", instance, "--seed", seed, "--trace", trace_file] + cand_params)
#print(command, file=errf)
return_code = subprocess.call(command, shell=True, stdout = outf, stderr = errf)

outf.close()
errf.close()

if return_code != 0:
    if os.path.isfile(err_file):
        with open(err_file, "r") as errf:
            sys.stderr.write(errf.read())
    target_runner_error("the command above exited with error code!")
    
if not os.path.isfile(out_file):
    print(command)
    target_runner_error("output file "+ out_file  +" not found!")
    

cost=[line.rstrip('\n') for line in open(out_file)][-10]
print(cost)
sys.exit(0)
