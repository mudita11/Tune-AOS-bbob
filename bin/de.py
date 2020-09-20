#!/usr/bin/env python3
import sys
import math
import numpy as np
import aos

class StatsFile():

    def __init__(self, filename):
        if filename and filename != "":
            self._file = open(filename, "w")
        self._dim = dim
        self._optimum = optimum

class TraceFile():
    # There are 51 targets equidistant between 1e2 and 1e-08 for each
    # problem instance.
    # _targets = 10**np.linspace(2,-8, 51, endpoint=True)
    # targets = [10**i for i in np.arange(2, -8.1, -0.2)] is used by coco (taken from coco code)
    # We increase the targets so that bad algorithms get a positive value
    _targets = 10**np.linspace(4,-8, 1 + 5*(4+8), endpoint=True)
    
    def __init__(self, filename, dim, optimum):
        self._file = None
        if filename:
            print(f"Writing trace file {filename}")
            self._file = open(filename, "w")
                    
        self._dim = dim
        self._optimum = optimum

    def print(self, fevals, bsf, header = False):
        if not self._file: return
        if header:
            self._file.write(f"# fevals/dim | frac | F - F_opt ({self._optimum}) | best fitness | fevals\n")
        fevalsdim = float(fevals) / self._dim
        error = bsf - self._optimum
        # FIXME This is failing!
        #assert error >= 0.0
        frac = np.sum(error <= self._targets) / float(len(self._targets))
        self._file.write(f"{fevalsdim} {frac} {error} {bsf} {fevals}\n")

    def __del__(self):
        if self._file:
            self._file.close()
            self._file = None
        

# FIXME: JADE adaptation speed c=0.1 ?
DE_params = {
        'FF':       [float, 0.5,    [0.1, 2.0],     'Scaling factor'],
        'CR':       [float, 1.0,    [0.1, 1.0],     'Crossover rate'],
        'NP':       [int,   200,    [50, 400],      'Population size'],
        'top_NP':   [float, 0.05,   [0.02, 1.0],    'Top candidates'],
        'mutation': [object, "DE/rand/1",
                     # This has to match the list mutation_names below.
                     # FIXME: Use that list to build this list.
                     ["DE/rand/1", "DE/rand/2", "DE/rand-to-best/2", "DE/current-to-rand/1", "DE/current_to_pbest", "DE/current_to_pbest_archived", "DE/best/1", "DE/current_to_best/1", "DE/best/2", "random", "aos"],
                     "Mutation strategy"]
        }


def DE_add_arguments(parser):
    group = parser.add_argument_group(title="DE parameters")
    for key, value in DE_params.items():
        if value[0] is object:
            group.add_argument('--' + key, default=value[1], choices=value[2], help=value[3])
        else:
            group.add_argument('--' + key, type=value[0], default=value[1], help=value[3])
            
def DE_irace_parameters(override = {}):
    output = "\n# DE parameters\n"
    for key, value in DE_params.items():
        output += aos.irace_parameter(key, value[0], value[2], help=value[3], override = override)
    return output
    
# NP: popsize
# F1 child fitness
# X parent population
# u offspring pop
# best: best candidate in current pop
# f_min = fitness minimum
# x_min = position minimum

# FIXME: Make DE a class and create a DE_mutation class, so each mutation is a
# subclass and we can list all subclasses like with do with AOS components in
# aos.py

def DE(fun, x0, lbounds, ubounds, budget, instance, instance_best_value,
       results_folder, stats_filename,
       FF, CR, NP, top_NP, mutation,
       OM_choice, rew_choice, rew_args, qual_choice, qual_args, prob_choice, prob_args, select_choice, select_args, known_aos):

    def rand1(population, samples, best, scale, NP, F, union):
        "DE/rand/1"
        r0, r1, r2 = samples[:3]
        return (population[r0] + scale * (population[r1] - population[r2]))

    def rand2(population, samples, best, scale, NP, F, union):
        "DE/rand/2"
        r0, r1, r2, r3, r4 = samples[:5]
        return (population[r0] + scale * (population[r1] - population[r2] + population[r3] - population[r4]))

    def rand_to_best2(population, samples, best, scale, NP, F, union):
        '''DE/rand-to-best/2'''
        r0, r1, r2, r3, r4 = samples[:5]
        return (population[r0] + scale * (population[best] - population[r0] + population[r1] - population[r2] + population[r3] - population[r4]))

    def current_to_rand1(population, samples, best, scale, NP, F, union):
        '''DE/current-to-rand/1'''
        r0, r1, r2 = samples[:3]
        # FIXME: i (current) should be a parameter.
        return (population[i] + scale * (population[r0] - population[i] + population[r1] - population[r2]))
    
    def current_to_pbest(population, samples, best, scale, NP, F, union):
        "DE/current_to_pbest"
        # '''Jade: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5208221'''
        #'''current to pbest (JADE-without archive)'''
        # MUDITA_check: In the following line top_best_index is collection of index representing top_NP number of best candidates from current parent population.
        # We use ceil so we never get 0.
        n = math.ceil(top_NP * NP)
        #print(f"{n}: {F}")
        #top_best_index_2 = [idx for (idx,v) in sorted(enumerate(F), key = lambda x: x[1])[:n]]
        top_best_index = np.argpartition(F, n)[:n]
        #assert np.all(np.sort(top_best_index) == np.sort(top_best_index_2)),f"{top_best_index} == {top_best_index_2}"
        rtop = np.random.choice(top_best_index)
        r0, r1 = samples[:2]
        return (population[i] + scale * (population[rtop] - population[i] + population[r0] - population[r1]))
    
    def current_to_pbest_archived(population, samples, best, scale, NP, F, union):
        "DE/current_to_pbest_archived"
        # '''Jade: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5208221'''
        # '''current to pbest (JADE-with archive)'''
        # We use ceil so that we never get 0.
        n = math.ceil(top_NP * NP)
        #print(f"{n}: {F}")
        #top_best_index_2 = [idx for (idx,v) in sorted(enumerate(F), key = lambda x: x[1])[:n]]
        top_best_index = np.argpartition(F, n)[:n]
        #assert np.all(np.sort(top_best_index) == np.sort(top_best_index_2)),f"{top_best_index} == {top_best_index_2}"
        rtop = np.random.choice(top_best_index)
        r0 = samples[0]
        return (population[i] + scale * (population[rtop] - population[i] + population[r0] - union[np.random.randint(NP)]))
        
    def best1(population, samples, best, scale, NP, F, union):
        "DE/best/1"
        r0, r1 = samples[:2]
        return (population[best] + scale * (population[r0] - population[r1]))
    
    def current_to_best1(population, samples, best, scale, NP, F, union):
        "DE/current_to_best/1"
        r0, r1 = samples[:2]
        return population[i] + scale * (population[best] - population[i] + population[r0] - population[r1])
    
    def best2(population, samples, best, scale, NP, F, union):
        '''DE/best/2'''
        r0, r1, r2, r3 = samples[:4]
        return population[best] + scale * (population[r0] - population[r1] + population[r2] - population[r3])
    
    def select_samples(popsize, candidate, number_samples):
        """
        obtain random integers from range(popsize),
        without replacement.  You can't have the original candidate either.
        """
        idxs = np.setdiff1d(np.arange(popsize), candidate)
        return np.random.choice(idxs, number_samples, replace = False)

    mutations = [rand1, rand2, rand_to_best2, current_to_rand1, current_to_pbest, current_to_pbest_archived, best1, current_to_best1, best2]
    # The names are given by the documentation of each function
    mutations_names = [ x.__doc__ for x in mutations ]
    n_operators = len(mutations)
    
    NP = int(NP)
    
    if mutation == "aos":
        if known_aos:
            aos_method = aos.AOS.build_known_AOS(
                known_aos, NP, budget, n_ops = n_operators,
                rew_args = rew_args, qual_args = qual_args,
                prob_args = prob_args, select_args = select_args)
        else:
            aos_method = aos.AOS(NP, budget, n_ops = n_operators, OM_choice = OM_choice,
                                 rew_choice = rew_choice, rew_args = rew_args,
                                 qual_choice = qual_choice, qual_args = qual_args,
                                 prob_choice = prob_choice, prob_args = prob_args,
                                 select_choice = select_choice, select_args = select_args)
        select_mutation = aos_method.select_operator
    elif mutation == "random":
        # lambda that returns a random integer
        select_mutation = lambda : np.random.randint(n_operators) 
    elif mutation in mutations_names:
        mutation = mutations_names.index(mutation)
        # lambda that always returns the same index
        select_mutation = lambda : mutation
    else:
        raise ValueError(f"unknown mutation {mutation}")

    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim = len(lbounds)
    # Initial population
    X = lbounds + (ubounds - lbounds) * np.random.rand(NP, dim)
    X[0, :] = x0
    # Evaluate population
    Fitness = np.apply_along_axis(fun, 1, X)
    best = np.argmin(Fitness)
    x_min, f_min = X[best, :], Fitness[best]
    
    u = np.full((NP,dim), 0.0)
    generation = 0

    trace_filename = None
    if results_folder:
        trace_filename = results_folder + f'/trace_{fun.id}.txt'
    trace = TraceFile(trace_filename, dim = dim, optimum = instance_best_value)
    #trace.print(1, Fitness[0], header = True)
    # We did NP fevals, remove them, then add as many as the number
    # needed to reach the best one (+1 because best is 0-based).
    trace.print(fun.evaluations - NP + best + 1, f_min, header = True)
    
    # MUDITA_check: archive is a mutation concept taken from Jade strategy (link to the paper given above mutation startegy current_to_pbest_archived). Its shape is initialised to be (budget + pop_size, dim) as nap values. Everytime a new parent population evolves, this population is put on archive from range(0:pop_size). Rest of the rows with nans in archive are replaced as worse parents are found compared to offspring, denoted as poor_candidates. Before this archive is used by Jade archived mutation startegy, a new list is created named union with all the non-nan rows from archive. Also if rows in union are more than pop_size, rows are randomly removed.
    archive = np.full(((budget+NP), dim), np.nan)
    # Operator used for each child
    opu = np.full(NP, -1)

    #statistics_file = open('si_vs_fe', 'a+')
    #statistics_file.write(str(fun)+'\n')
    while fun.evaluations + NP <= budget and not fun.final_target_hit:
        fill_points = np.random.randint(dim, size = NP)
        # MUDITA_check: Following 4 lines are added to implement above explained idea. Basically, top pop_size rows are replaced by cuurent parent population. non-nan union is created and its len of pop_size is maintained.
        archive[:NP] = X
        union = archive[~np.isnan(archive[:,0])]
        if len(union) > NP:
            union = union[np.random.randint(len(union), size = NP), :]
    
        for i in range(NP):
            # No mutation strategy needs more than 5.
            r = select_samples(NP, i, 5)
            SI = select_mutation()
            assert SI >= 0 and SI <= len(mutations)
            opu[i] = SI
            #statistics_file.write(str(SI)+'\n')
            bprime = mutations[SI](X, r, best, FF, NP, Fitness, union)
            # binomial crossover
            crossovers = (np.random.rand(dim) < CR)
            # fill_points makes sure that at least one position comes from
            # bprime.
            crossovers[fill_points[i]] = True
            u[i,:] = np.where(crossovers, bprime, X[i, :])

        # Evaluate the child population
        F1 = np.apply_along_axis(fun, 1, u)

        # We need to call OM_update after evaluating F1 and before updating
        # F_bsf, otherwise there will never be an improvement.
        if mutation == "aos":
            aos_method.OM_Update(Fitness, F1, F_bsf = f_min, opu = opu)
        
        # Find the best child.
        best = np.argmin(F1)
        if F1[best] < f_min:
            x_min, f_min = u[best, :], F1[best]
            # We did NP fevals, remove them, then add as many as the number
            # needed to reach the best one (+1 because best is 0-based).
            trace.print(fun.evaluations - NP + best + 1, f_min)
    
        # MUDITA_check: Following 4 lines append the archive with worse parents than offsprings.
        # Maintaining archive (a list)
        start = np.argwhere(np.isnan(archive[:, 0]))[0][0]
        poor_chandidates = X[Fitness > F1]
        end = start+len(poor_chandidates)
        archive[start:end] = poor_chandidates
    
        # Replace parent if their child improves them.
        Fitness = np.where(F1 <= Fitness, F1, Fitness)
        X[F1 <= Fitness, :] = u[F1 <= Fitness, :]

        generation += 1

    if mutation == "aos" and results_folder:
        genwindow_filename = results_folder + f'/genw_{fun.id}.txt.gz'
        print(f"Writing gen window to {genwindow_filename}")
        aos_method.gen_window.write_to(genwindow_filename)
            
    #statistics_file.close()

    return f_min

