#!/usr/bin/env python

try: range = xrange
except NameError: pass
import os, sys
import time
import numpy as np  # "pip install numpy" installs numpy
import cocoex
from cocoex import Suite, Observer, log_level
import random
import math
import csv
from numpy.linalg import inv
#import uuid

import aos

def DE_add_arguments(parser):
    group = parser.add_argument_group(title="DE parameters")
    group.add_argument('--FF', type=float, default=0.5, help='Scaling factor (DE parameter)')
    group.add_argument('--CR', type=float, default=1.0, help='Crossover rate (DE parameter)')
    group.add_argument('--NP', type=int, default=200, help='Population size (DE parameter)')

def DE_irace_parameters():
    output = "\n# DE parameters\n"
    output += aos.irace_parameter("FF", float, [0.1, 2.0], help = 'Scaling factor (DE parameter)')
    output += aos.irace_parameter("CR", float, [0.1, 1.0], help = 'Crossover rate (DE parameter)')
    output += aos.irace_parameter("NP", int,   [50, 400],  help='Population size (DE parameter)')
    return output
    
# NP: popsize
# F1 child fitness
# X parent population
# u offspring pop
# index: best candidate in current pop
# f_min = fitness minimum
# x_min = position minimum

def DE(fun, lbounds, ubounds, budget, instance, instance_best_value,
       trace_filename, stats_filename,
       FF, CR, NP,
       OM_choice, rew_choice, rew_args, qual_choice, qual_args, prob_choice, prob_args, select_choice):

        
    def rand1(population, samples, best, scale):
        """DE/rand/1"""
        r0, r1, r2 = samples[:3]
        return (population[r0] + scale * (population[r1] - population[r2]))

    def rand2(population, samples, best, scale): # DE/rand/2
        r0, r1, r2, r3, r4 = samples[:5]
        return (population[r0] + scale * (population[r1] - population[r2] + population[r3] - population[r4]))

    def rand_to_best2(population, samples, best, scale): # DE/rand-to-best/2
        r0, r1, r2, r3, r4 = samples[:5]
        return (population[r0] + scale * (population[best] - population[r0] + population[r1] - population[r2] + population[r3] - population[r4]))

    def current_to_rand1(population, samples, best, scale): # DE/current-to-rand/1
        r0, r1, r2 = samples[:3]
        return (population[i] + scale * (population[r0] - population[i] + population[r1] - population[r2]))

    def select_samples(popsize, candidate, number_samples):
        """
        obtain random integers from range(popsize),
        without replacement.  You can't have the original candidate either.
        """
        idxs = list(range(popsize))
        idxs.remove(candidate)
        return(np.random.choice(idxs, 5, replace = False))

    NP = int(NP)
    opu = np.full(NP, -1)
    
    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim, x_min, f_min = len(lbounds), (lbounds + ubounds) / 2, None
    X = lbounds + (ubounds - lbounds) * np.random.rand(NP, dim)
    u = np.full((NP,dim), 0)

    F = np.apply_along_axis(fun, 1, X)
    #F = [fun(x) for x in X]
    max_budget = budget
    budget -= NP
    
    generation = 0
    index = np.argmin(F);
    # MANUEL: f_min is None!
    if f_min is None or F[index] < f_min:
        x_min, f_min = X[index, :], F[index];
    best_so_far = f_min
    best_so_far1 = best_so_far

    mutations = [rand1, rand2, rand_to_best2, current_to_rand1]
    n_operators = len(mutations)
    
    # Test different combinations
    aos_method = aos.Unknown_AOS(NP, n_ops = n_operators, OM_choice = OM_choice,
                                 rew_choice = rew_choice, rew_args = rew_args,
                                 qual_choice = qual_choice, qual_args = qual_args,
                                 prob_choice = prob_choice, prob_args = prob_args,
                                 select_choice = select_choice)

    stats_file = None
    if stats_filename:
        stats_file = open(stats_filename, 'w+')
    #problem_data = "i" + "-" + str(uuid.uuid4()) + "-" + str(instance) + ".txt"

    trace_file = None
    if trace_filename:
        trace_file = open(trace_filename, "w")
        trace_file.write("%fevals"+" "+" error"+" "+" best"+"\n")
    # MANUEL: Where do these numbers come from????
    # MUDITA: There are 51 targets equidistant between 1e2 and 1e-08 for each problem instance.
    #    target_diff = (1e-8 - 1e2 +1)/ 51
    #    target = 1e+2
    error = best_so_far - instance_best_value
    #if error <= target:
        #print(budget, error, best_so_far)
        ## MANUEL: Is this exactly what BBOB is doing?
        ## MUDITA: The numbers corresponding to specific targets generated by our DE algorithm match with the number corresponding to targets generated by bbob code. However, bbob is writing data for more targets. Not sure if they are using all that data to generate graphs.
    ## MANUEL: generation is zero
    if trace_file:
        trace_file.write(str((generation*NP) + index)+" "+str(error)+" "+str(best_so_far)+"\n")

    
    while budget > 0:
        
        fill_points = np.random.randint(dim, size = NP)
        
        for i in range(NP):
            SI = aos_method.select_operator()
            if stats_file:
                stats_file.write("{} {}\n".format(generation, SI))
            assert SI >= 0 and SI <= len(mutations)
            mutate = mutations[SI]
            opu[i] = SI
            # No mutation strategy needs more than 5.
            r = select_samples(NP, i, 5)
            best = np.argmin(F)
            crossovers = (np.random.rand(dim) < CR)
            crossovers[fill_points[i]] = True
            # MANUEL: What is this trial?
            # trial = aos_method.X[i]
            bprime = mutate(X, r, best, FF)
            u[i,:] = np.where(crossovers, bprime, X[i, :])
    
        F1 = np.apply_along_axis(fun, 1, u)
        aos_method.OM_Update(F, F1, F_bsf = best_so_far, opu = opu)
                
        #output_file.write(str(aos_method.reward)+"\n")
        #output_file.write(str(aos_method.quality)+"\n")
        #output_file.write(str(aos_method.probability)+"\n")
        #fitness_swap = [a<p for a,p in zip(F1,F)]
        F = np.where(F1 <= F, F1, F)
        X[F1 <= F, :] = u[F1 <= F, :]
        
        index = np.argmin(F)
        if f_min is None or F[index] < f_min:
            x_min, f_min = X[index, :], F[index]
            best_so_far1 = f_min;
        if best_so_far1 < best_so_far:
            best_so_far = best_so_far1
            error = best_so_far - instance_best_value
            # MANUEL: generation is still zero for the first iteration
            if trace_file:
                trace_file.write(str((generation*NP) + index)+" "+str(error)+" "+str(best_so_far)+"\n")

        generation += 1
        budget -= NP
    #output_file.write("Last generation number"+str(generation)+".................................................................................\n")
    #output_file.close()
    #print("one configuartion tested on one instance")
    #file.close()
    return best_so_far

