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
import numpy as np

max_obj = 8.977281728e+01
# fevals * dimension f-evaluations
fevals = 1000
# Options are: suite_name fevals batch total_batches
exe = "python3 ../bin/DE_AOS.py bbob {} 1 1".format(fevals)

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
instance = sys.argv[4]
cand_params = sys.argv[5:]

#cand_params = ['--FF', '0.3', '--CR', '0.2', '--NP', '50', '--OM_choice', '1', '--rew_choice', '0', '--qual_choice', '2', '--prob_choice', '0', '--select_choice', '1', '--fix_appl', '10', '--adaptation_rate', '0.57', '--p_min', '0.05', '--error_prob','0.1']

#cand_params=[str(c1), str(c2), str(c3), str(c4), str(c5), str(c6), str(c7), str(c8), str(c9), str(c10), str(c11), str(c12), str(c13), str(c14), str(c15), str(c16), str(c17), str(c18), str(c19), str(c20), str(c21), str(c22), str(c23), str(c24), str(c25), str(c26), str(c27), str(c28), str(c29), str(c30), str(c31), str(c32)]
#print("cand_params", cand_params)
trace_file = "trace_" + str(candidate_id) + "-" + str(instance_id) + ".txt"

# Define the stdout and stderr files.
out_file = "c" + str(candidate_id) + "-" + str(instance_id) + ".stdout"
err_file = "c" + str(candidate_id) + "-" + str(instance_id) + ".stderr"

# Build the command, run it and save the output to a file,
# to parse the result from it.
# 
# Stdout and stderr files have to be opened before the call().
#
# Exit with error if something went wrong in the execution.
command = " ".join([exe, "-i", instance, "--seed", seed, "--trace", trace_file] + cand_params)
#print("command",command)

outf = open(out_file, "w")
errf = open(err_file, "w")
print(command, file=errf)
return_code = subprocess.call(command, shell=True, stdout = outf, stderr = errf)
outf.close()
errf.close()

if return_code != 0:
    print(command)
    now = datetime.datetime.now()
    print(str(now) + " error: command returned code " + str(return_code))
    sys.exit(1)

if not os.path.isfile(out_file):
    print(command)
    now = datetime.datetime.now()
    print(str(now) + " error: output file "+ out_file  +" not found.")
    sys.exit(1)

# FIXME: We cannot normalize per dataset, we need to include an upper bound of
# fevals and fitness.
points = np.loadtxt(trace_file, comments = "%", usecols = (0,1))
points[:, 0] = points[:, 0] / fevals
if (np.max(points[:,1]) - np.min(points[:,1])) != 0:
    points[:,1] = (points[:,1] - np.min(points[:,1])) / (np.max(points[:,1]) - np.min(points[:,1]))
else:
    points[:,1] = 1

# See README.txt to install this
from pygmo import hypervolume

ref_point = [1.1, 1.1]
hv = hypervolume(points)
cost = hv.compute(ref_point)
#cost=[line.rstrip('\n') for line in open(out_file)][-8]

# This is an example of reading a number from the output.
# It assumes that the objective value is the first number in
# the first column of the last line of the output.
# from http://stackoverflow.com/questions/4703390

# print("Cost:= ",cost)
print(-cost)

sys.exit(0)
#print("End of target-runner")
