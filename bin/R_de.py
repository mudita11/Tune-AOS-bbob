#!/usr/bin/env python

try: range = xrange
except NameError: pass
import sys
import numpy as np
import aos

class TraceFile():
    # There are 51 targets equidistant between 1e2 and 1e-08 for each
    # problem instance.
    # _targets = 10**np.linspace(2,-8, 51, endpoint=True)
    # targets = [10**i for i in np.arange(2, -8.1, -0.2)] is used by coco (taken from coco code)
    # We increase the targets so that bad algorithms get a positive value
    _targets = 10**np.linspace(4,-8, 1 + 5*(4+8), endpoint=True)
    _file = None

    def __init__(self, filename, dim, optimum):
        if filename and filename != "":
            self._file = open(filename, "w")
        self._dim = dim
        self._optimum = optimum

    def print(self, fevals, bsf, header = False):
        if not self._file:
            return
        if header:
            self._file.write("% fevals/dim | frac | F - F_opt ({}) | best fitness | fevals\n".format(self._optimum))
        fevalsdim = float(fevals) / self._dim
        error = bsf - self._optimum
        assert error >= 0.0
        frac = np.sum(error <= self._targets) / float(len(self._targets))
        self._file.write("{0} {1} {2} {3} {4}\n".format(
            fevalsdim, frac, error, bsf, fevals))

    def close(self, ):
        if self._file:
            _file.close()

            
DE_params = {
        'FF':       [float, 0.5,    [0.1, 2.0],     'Scaling factor'],
        'CR':       [float, 1.0,    [0.1, 1.0],     'Crossover rate'],
        'NP':       [int,   200,    [50, 400],      'Population size'],
        'top_NP':   [float, 0.05,   [0.02, 1.0],    'Top candidates'],
        'mutation': [object, "DE/rand/1",
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

def initialise_evaluate(lbounds, ubounds, NP, budget, dim, fun, x0, x_min=None):
    # Initialise population
    X = lbounds + (ubounds - lbounds) * np.random.rand(NP, dim)
    X[0, :] = x0
    if x_min is not None:
        X[1, :] = x_min
    # Evaluate population
    F = np.apply_along_axis(fun, 1, X)
    best = np.argmin(F)
    x_min, f_min = X[best, :], F[best]
    u = np.full((NP,dim), 0.0)
    archive = np.full(((budget+NP), dim), np.nan)
    union = np.copy(archive)
    return X, F, best, x_min, f_min, u, archive, union

def DE(fun, x0, lbounds, ubounds, budget, instance, instance_best_value,
       trace_filename, stats_filename,
       FF, CR, NP, top_NP, mutation,
       OM_choice, rew_choice, rew_args, qual_choice, qual_args, prob_choice, prob_args, select_choice, select_args):

    def rand1(population, samples, best, scale, NP, F, union, x_min):
        """DE/rand/1"""
        r0, r1, r2 = samples[:3]
        return (population[r0] + scale * (population[r1] - population[r2]))

    def rand2(population, samples, best, scale, NP, F, union, x_min):
        '''DE/rand/2'''
        r0, r1, r2, r3, r4 = samples[:5]
        return (population[r0] + scale * (population[r1] - population[r2] + population[r3] - population[r4]))

    def rand_to_best2(population, samples, best, scale, NP, F, union, x_min):
        '''DE/rand-to-best/2'''
        r0, r1, r2, r3, r4 = samples[:5]
        return (population[r0] + scale * (population[best] - population[r0] + population[r1] - population[r2] + population[r3] - population[r4]))

    def current_to_rand1(population, samples, best, scale, NP, F, union, x_min):
        '''DE/current-to-rand/1'''
        r0, r1, r2 = samples[:3]
        return (population[i] + scale * (population[r0] - population[i] + population[r1] - population[r2]))
    
    def current_to_pbest(population, samples, best, scale, NP, F, union, x_min):
        '''current to pbest (JADE-without archive)'''
        r0, r1 = samples[:2]
        #percent_population = int(0.05 * NP)
        top_best_index = [idx for (idx,v) in sorted(enumerate(F), key = lambda x: x[1])[:int(top_NP * NP)]]
        return (population[i] + scale * (population[np.random.choice(top_best_index)] - population[i] + population[r0] - population[r1]))
    
    def current_to_pbest_archived(population, samples, best, scale, NP, F, union, x_min):
        '''current to pbest (JADE-with archive)'''
        r0 = samples[:1]
        top_best_index = [idx for (idx,v) in sorted(enumerate(F), key = lambda x: x[1])[:int(top_NP * NP)]]
        return (population[i] + scale * (population[np.random.choice(top_best_index)] - population[i] + population[r0] - union[np.random.randint(NP)]))
    
    def current_to_best1(population, samples, best, scale, NP, F, union, x_min):
        '''DE/current-to-best/1'''
        r0, r1 = samples[:2]
        return (population[i] +scale * (population[best] - population[i] + population[r0] - population[r1]))
    
    def best1(population, samples, best, scale, NP, F, union, x_min):
        '''DE/best/1'''
        r0, r1 = samples[:2]
        return (population[best] + scale * (population[r0] - population[r1]))
    
    def best2(population, samples, best, scale, NP, F, union, x_min):
        '''DE/best/2'''
        r0, r1, r2, r3 = samples[:4]
        return (population[best] + scale * (population[r0] - population[r1] + population[r2] - population[r3]))
    
    def select_samples(popsize, candidate, number_samples):
        """
        obtain random integers from range(popsize),
        without replacement.  You can't have the original candidate either.
        """
        idxs = list(range(popsize))
        idxs.remove(candidate)
        return(np.random.choice(idxs, number_samples, replace = False))

    mutations = [rand1, rand2, rand_to_best2, current_to_rand1, current_to_pbest, current_to_pbest_archived, best1, current_to_best1, best2]
    mutations_names = [ x.__doc__ for x in mutations]
    n_operators = len(mutations)
    
    NP = int(NP)
    opu = np.full(NP, -1)
    
    if mutation == "aos":
        aos_method = aos.Unknown_AOS(NP, budget, n_ops = n_operators, OM_choice = OM_choice,
                                     rew_choice = rew_choice, rew_args = rew_args,
                                     qual_choice = qual_choice, qual_args = qual_args,
                                     prob_choice = prob_choice, prob_args = prob_args,
                                     select_choice = select_choice, select_args = select_args)
        select_mutation = aos_method.select_operator
    elif mutation == "random":
        select_mutation = lambda : np.random.randint(n_operators) 
    elif mutation in mutations_names:
        mutation = mutations_names.index(mutation)
        select_mutation = lambda : mutation
    else:
        raise ValueError("unknown mutation " + mutation)

    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim = len(lbounds)
    
    X, F, best, x_min, f_min, u, archive, union = initialise_evaluate(lbounds, ubounds, NP, budget, dim, fun, x0)
    #stats_file = None
    #if stats_filename:
        #stats_file = open(stats_filename, 'w+')
    
    stagnation_count = 0
    generation = 0
    #trace = TraceFile(trace_filename, dim = dim, optimum = instance_best_value)
    #trace.print(1, F[0], header = True)
    # We did NP fevals, remove them, then add as many as the number
    # needed to reach the best one (+1 because best is 0-based).
    #trace.print(fun.evaluations - NP + best + 1, f_min)

    #statistics_file = open('si_vs_fe', 'a+')
    #statistics_file.write(str(fun)+'\n')
    while fun.evaluations + NP <= budget and not fun.final_target_hit:
        max_X = np.max(X, axis = 1)
        min_X = np.min(X, axis = 1)
        max_F = np.max(F)
        min_F = np.min(F)
        if (np.any((max_X - min_X) < (1e-12 * np.fabs(max_X)))) or ((max_F - min_F) < (1e-12 * np.fabs(max_F))) or (stagnation_count >= 500*dim):
            print("provoke ", generation, ((max_X - min_X) < (1e-12 * np.fabs(max_X))), (np.any(max_F - min_F) < (1e-12 * np.fabs(max_F))), (stagnation_count >= 500*dim))
            X, F, best, x_min, f_min, u, archive, union = initialise_evaluate(lbounds, ubounds, NP, budget, dim, fun, x0, x_min)
            stagnation_count = 0
        
        fill_points = np.random.randint(dim, size = NP)
        archive[:NP] = X
        union = np.copy(archive)
        union = union[~np.isnan(union[:,0])]
        if len(union) > NP:
            union = union[np.random.randint(len(union), size = NP), :]
        
        for i in range(NP):
            # No mutation strategy needs more than 5.
            r = select_samples(NP, i, 5)
            crossovers = (np.random.rand(dim) < CR)
            crossovers[fill_points[i]] = True
            SI = select_mutation()
            assert SI >= 0 and SI <= len(mutations)
            opu[i] = SI
            #statistics_file.write(str(SI)+'\n')
            mutate = mutations[SI]
            bprime = mutate(X, r, best, FF, NP, F, union, x_min)
            u[i,:] = np.where(crossovers, bprime, X[i, :])

        # Evaluate the child population
        F1 = np.apply_along_axis(fun, 1, u)

        # Find the best child.
        best = np.argmin(F1)
        if F1[best] < f_min:
            x_min, f_min = u[best, :], F1[best]
            stagnation_count = 0
        else:
            stagnation_count += NP
        # We did NP fevals, remove them, then add as many as the number
        # needed to reach the best one (+1 because best is 0-based).
        #trace.print(fun.evaluations - NP + best + 1, f_min)

        if mutation == "aos":
            aos_method.OM_Update(F, F1, F_bsf = f_min, opu = opu)
    
        # Maintainng archive (a list)
        start = np.argwhere(np.isnan(archive[:,0]))[0][0]
        poor_chandidates = X[F>F1]
        end = start+len(poor_chandidates)
        archive[start:end] = poor_chandidates
    
        # Replace parent if their child improves them.
        F = np.where(F1 <= F, F1, F)
        X[F1 <= F, :] = u[F1 <= F, :]

        generation += 1

    if mutation == "aos":
        aos_method.gen_window.write_to(sys.stderr)
    #statistics_file.close()

    return f_min

