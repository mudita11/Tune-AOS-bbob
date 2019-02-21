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

# chunk is popsize
# F1 child fitness
# X parent population
# u offspring pop
# index: best candidate in current pop
# f_min = fitness minimum
# x_min = position minimum

def DE(fun, lbounds, ubounds, budget, instance,
       trace_file,
<<<<<<< HEAD
       FF, CR, NP, #alpha, W, phi, max_gen, C, c1_quality6, c2_quality6, gamma, delta, decay_reward3, decay_reward4, int_a_reward5, b_reward5, e_reward5, a_reward71, c_reward9, int_b_reward9, int_a_reward9, int_a_reward101, b_reward101, instance_best_value,
       OM_choice, rew_choice, rew_args, qual_choice, qual_args, prob_choice, prob_args, select_choice):
=======
       FF, CR, alpha, W, phi, max_gen, C, c1_quality6, c2_quality6, gamma, delta, decay_reward3, decay_reward4, int_a_reward5, b_reward5, e_reward5, a_reward71, c_reward9, int_b_reward9, int_a_reward9, int_a_reward101, b_reward101, instance_best_value,
       prob_choice, prob_args):
>>>>>>> 9689958891cbb8e0fec6df86d202c39f597b03a3
    
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



    generation = 0
    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim, x_min, f_min = len(lbounds), (lbounds + ubounds) / 2, None
    NP = 10 * dim
    chunk = NP
    X = lbounds + (ubounds - lbounds) * np.random.rand(chunk, dim)
    F = [fun(x) for x in X];
    max_budget = np.copy(budget)
    budget -= chunk
    
    u = [[0 for j in range(int(dim))] for i in range(int(chunk))]
    F1 = np.zeros(int(chunk));
    
    index = np.argmin(F);
    if f_min is None or F[index] < f_min:
        x_min, f_min = X[index], F[index];
    best_so_far = f_min
    best_so_far1 = best_so_far

    mutations = [rand1, rand2, rand_to_best2, current_to_rand1]
    n_operators = len(mutations)
    
    # Test different combinations
<<<<<<< HEAD
#    OO = 2 # Range from 1-7
#    RR = 7 # Range from 0-12; window: {3, 4, 8} and max_gen: {5, 6, 7, 9, 10, 11}.
#    QQ = 4 # Range from 0-5
#    SS = 0 # Range from 0-1
    aos_method = aos.Unknown_AOS(chunk, F1, F, u, X, f_min, x_min, best_so_far, best_so_far1,
                                 #OO, RR, QQ, SS, n_ops = n_operators, adaptation_rate = alpha, phi = phi, max_gen = max_gen, scaling_factor = C, c1_quality6 = c1_quality6, c2_quality6 = c2_quality6, discount_rate = gamma, delta = delta, decay_reward3 = decay_reward3, decay_reward4 = decay_reward4,  int_a_reward5 = int_a_reward5, b_reward5 = b_reward5, e_reward5 = e_reward5, a_reward71 = a_reward71, c_reward9 = c_reward9, int_b_reward9 = int_b_reward9, int_a_reward9 = int_a_reward9, int_a_reward101 = int_a_reward101, b_reward101 = b_reward101, window_size = W,
        OM_choice = OM_choice, rew_choice = rew_choice, rew_args = rew_args, qual_choice = qual_choice, qual_args = qual_args, prob_choice = prob_choice, prob_args = prob_args, select_choice = select_choice)
=======
    OO = 2 # Range from 1-7
    RR = 7 # Range from 0-12; window: {3, 4, 8} and max_gen: {5, 6, 7, 9, 10, 11}.
    QQ = 4 # Range from 0-5
    SS = 0 # Range from 0-1
    aos_method = aos.Unknown_AOS(chunk, F1, F, u, X, f_min, x_min, best_so_far, best_so_far1, OO, RR, QQ, SS, n_ops = n_operators, adaptation_rate = alpha, phi = phi, max_gen = max_gen, scaling_factor = C, c1_quality6 = c1_quality6, c2_quality6 = c2_quality6, discount_rate = gamma, delta = delta, decay_reward3 = decay_reward3, decay_reward4 = decay_reward4,  int_a_reward5 = int_a_reward5, b_reward5 = b_reward5, e_reward5 = e_reward5, a_reward71 = a_reward71, c_reward9 = c_reward9, int_b_reward9 = int_b_reward9, int_a_reward9 = int_a_reward9, int_a_reward101 = int_a_reward101, b_reward101 = b_reward101, window_size = W,
                                 prob_choice = prob_choice, prob_args = prob_args)
>>>>>>> 9689958891cbb8e0fec6df86d202c39f597b03a3

    
    #output_file = open('output_statistics.txt', 'w+')
    #problem_data = "i" + "-" + str(uuid.uuid4()) + "-" + str(instance) + ".txt"
    
    inst_file = open(trace_file, "w")
    inst_file.write("%fevals"+" "+" error"+" "+" best"+"\n")
    # MANUEL: Where do these numbers come from????
<<<<<<< HEAD
    # MUDITA: There are 51 targets equidistant between 1e2 and 1e-08 for each problem instance.
=======
>>>>>>> 9689958891cbb8e0fec6df86d202c39f597b03a3
    target_diff = (1e-8 - 1e2 +1)/ 51
    target = 1e+2
    error = best_so_far - instance_best_value
    #print("E",error)
    if error <= target:
        #print(budget, error, best_so_far)
        ## MANUEL: Is this exactly what BBOB is doing?
<<<<<<< HEAD
        ## MUDITA: The numbers corresponding to specific targets generated by our DE algorithm match with the number corresponding to targets generated by bbob code. However, bbob is writing data for more targets. Not sure if they are using all that data to generate graphs.
=======
>>>>>>> 9689958891cbb8e0fec6df86d202c39f597b03a3
        inst_file.write(str((generation*NP) + index)+" "+str(error)+" "+str(best_so_far)+"\n")
    while budget > 0:
        
        fill_points = np.random.randint(dim, size = NP)
        
        for i in range(NP):
            SI = aos_method.Selection(); #output_file.write(str(SI)+"\n")
            assert SI >= 0 and SI <= len(mutations)
            mutate = mutations[SI]
            aos_method.opu[i] = SI
            # No mutation strategy needs more than 5.
            r = select_samples(NP, i, 5)
            best = np.argmin(aos_method.F)
            crossovers = (np.random.rand(dim) < CR)
            crossovers[fill_points[i]] = True
            # trial = aos_method.X[i]
            bprime = mutate(aos_method.X, r, best, FF)
            aos_method.u[i][:] = np.where(crossovers, bprime, aos_method.X[i])
    
        aos_method.F1 = [fun(np.array(x)) for x in aos_method.u]
        
        aos_method.OM_Update()
        #output_file.write(str(aos_method.reward)+"\n")
        #output_file.write(str(aos_method.quality)+"\n")
        #output_file.write(str(aos_method.probability)+"\n")
        #fitness_swap = [a<p for a,p in zip(F1,F)]
        for i in range(NP):
            if aos_method.F1[i] <= aos_method.F[i]:
                aos_method.F[i] = aos_method.F1[i]
                aos_method.X[i] = aos_method.u[i]
            
        #aos_method.X[i] = np.where(aos_method.F1 < aos_method.F, self.u[i], aos_method.X[i])
        index = np.argmin(aos_method.F)
        if aos_method.f_min is None or aos_method.F[index] < aos_method.f_min:
            aos_method.x_min, aos_method.f_min = aos_method.X[index], aos_method.F[index]
        aos_method.best_so_far1 = aos_method.f_min;
        if aos_method.best_so_far1 < aos_method.best_so_far:
            aos_method.best_so_far = aos_method.best_so_far1
        
        error = aos_method.best_so_far - instance_best_value
        #print("E",error)
        if error <= target:
            #print(budget, error, best_so_far)
            inst_file.write(str((generation*NP) + index)+" "+str(error)+" "+str(best_so_far)+"\n")
            target = target + target_diff
                    
        generation = generation + 1
        budget -= chunk
    #output_file.write("Last generation number"+str(generation)+".................................................................................\n")
    #output_file.close()
    #print("one configuartion tested on one instance")
    #file.close()
    return aos_method.best_so_far

