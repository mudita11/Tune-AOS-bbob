#!/bin/bash


fes=500000
de="./bin/DE_AOS.py bbob $fes 1 1"

# MANUEL: where is the output saved?
# Also, if you read the command-line parameters from a file, one per line, you can do something like:
# cat file | xargs -n 1 $de
# See http://man7.org/linux/man-pages/man1/xargs.1.html

# And if the computer you use has the 'parallel' command, then you can run in parallel with:
# cat file | parallel $de
# See https://www.gnu.org/software/parallel/man.html#EXAMPLE:-Working-as-xargs--n1.-Argument-appending



# tuned_rand_1_target_fe
$de --FF 0.36 --CR 0.21 --NP 50 --mutation DE/rand/1
# tuned_rand_1_error_fe
$de --FF 0.3 --CR 0.48 --NP 50 --mutation DE/rand/1
# tuned_rand_1_best
$de --FF 0.4 --CR 0.16 --NP 60 --mutation DE/rand/1

# tuned_rand_2_target_fe
$de --FF 0.19 --CR 0.3 --NP 51 --mutation DE/rand/2
# tuned_rand_1_error_fe
$de --FF 0.29 --CR 0.53 --NP 51 --mutation DE/rand/2
# tuned_rand_1_best
$de --FF 0.34 --CR 0.37 --NP 54 --mutation DE/rand/2

# tuned_rand_best_2_target_fe
$de --FF 0.5 --CR 0.31 --NP 58 --mutation DE/rand-to-best/2
# tuned_rand_best_2_error_fe
$de --FF 0.4 --CR 0.45 --NP 52 --mutation DE/rand-to-best/2
# tuned_rand_best_2_best
$de --FF 0.4 --CR 0.16 --NP 60 --mutation DE/rand-to-best/2

# tuned_current_rand_1_target_fe
$de --FF 0.63 --CR 0.21 --NP 50 --mutation DE/current-to-rand/1
# tuned_current_rand_1_error_fe
$de --FF 0.59 --CR 0.23 --NP 50 --mutation DE/current-to-rand/1
# tuned_current_rand_1_best
$de --FF 0.69 --CR 0.22 --NP 58 --mutation DE/current-to-rand/1
