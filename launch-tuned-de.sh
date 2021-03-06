#!/bin/bash


fes=5000
file="configurations-de.txt"

de="./bin/DE_AOS.py bbob $fes 1 1 --train_test test"

# MANUEL: where is the output saved?
# Also, if you read the command-line parameters from a file, one per line, you can do something like:
# cat file | xargs -n 1 $de
# See http://man7.org/linux/man-pages/man1/xargs.1.html
# cat $file | xargs -I{} -n 1 echo $de {}
cat $file | parallel --verbose --eta -j 16 --colsep ' ' "$de {} --result_folder de-run-{#} > de-run-{#}.output"

# And if the computer you use has the 'parallel' command, then you can run in parallel with:
# cat file | parallel $de
# See https://www.gnu.org/software/parallel/man.html#EXAMPLE:-Working-as-xargs--n1.-Argument-appending


