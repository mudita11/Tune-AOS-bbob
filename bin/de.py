#!/usr/bin/env python

try: range = xrange
except NameError: pass
import sys
import numpy as np  # "pip install numpy" installs numpy
import aos

class TraceFile():
    # There are 51 targets equidistant between 1e2 and 1e-08 for each
    # problem instance.
    #_targets = 10**np.linspace(2,-8, 51, endpoint=True)
    # We increase the targets so that bad algorithms get a positive value
    _targets = 10**np.linspace(4,-8, 1 + 5*(4+8), endpoint=True)
    _file = None

    def __init__(self, filename, dim, optimum):
        if filename != "":
            self._file = open(filename, "w")
        self._dim = dim
        self._optimum = optimum

    def print(self, fevals, bsf, header = False):
        if not self._file:
            return

        if header:
            self._file.write("% fevals/dim | frac | F - F_opt ({}) | best fitness | fevals\n".format(self._optimum))
        error = bsf - self._optimum
        frac = np.sum(error <= self._targets) / float(len(self._targets))
        self._file.write("{0} {1} {2} {3} {4}\n".format(
            float(fevals) / self._dim, frac, error, bsf, fevals))

    def close(self, ):
        if self._file:
            _file.close()

            
DE_params = {
        'FF': [float, 0.5, [0.1, 2.0],'Scaling factor'],
        'CR': [float, 1.0, [0.1, 1.0],'Crossover rate'],
        'NP': [int,   200, [50, 400], 'Population size'],
        'mutation': [object, "aos",
                     ["DE/rand/1","DE/rand/2","DE/rand/1","DE/rand-to-best/2", "random", "aos"],
                     "Mutation strategy"]
        }


def DE_add_arguments(parser):
    group = parser.add_argument_group(title="DE parameters")
    for key, value in DE_params.items():
        if value[0] is object:
            group.add_argument('--' + key, default=value[1], choices=value[2], help=value[3])
        else:
            group.add_argument('--' + key, type=value[0], default=value[1], help=value[3])
            
def DE_irace_parameters():
    output = "\n# DE parameters\n"
    for key, value in DE_params.items():
        output += aos.irace_parameter(key, value[0], value[2], help=value[3])
    return output
    
# NP: popsize
# F1 child fitness
# X parent population
# u offspring pop
# best: best candidate in current pop
# f_min = fitness minimum
# x_min = position minimum
    
def DE(fun, x0, lbounds, ubounds, budget, instance, instance_best_value,
       trace_filename, stats_filename,
       FF, CR, NP, mutation,
       OM_choice, rew_choice, rew_args, qual_choice, qual_args, prob_choice, prob_args, select_choice):

    def rand1(population, samples, best, scale):
        """DE/rand/1"""
        r0, r1, r2 = samples[:3]
        return (population[r0] + scale * (population[r1] - population[r2]))

    def rand2(population, samples, best, scale):
        '''DE/rand/2'''
        r0, r1, r2, r3, r4 = samples[:5]
        return (population[r0] + scale * (population[r1] - population[r2] + population[r3] - population[r4]))

    def rand_to_best2(population, samples, best, scale):
        '''DE/rand-to-best/2'''
        r0, r1, r2, r3, r4 = samples[:5]
        return (population[r0] + scale * (population[best] - population[r0] + population[r1] - population[r2] + population[r3] - population[r4]))

    def current_to_rand1(population, samples, best, scale):
        '''DE/current-to-rand/1'''
        r0, r1, r2 = samples[:3]
        return (population[i] + scale * (population[r0] - population[i] + population[r1] - population[r2]))

    def select_samples(popsize, candidate, number_samples):
        """
        obtain random integers from range(popsize),
        without replacement.  You can't have the original candidate either.
        """
        idxs = list(range(popsize))
        idxs.remove(candidate)
        return(np.random.choice(idxs, number_samples, replace = False))


    mutations = [rand1, rand2, rand_to_best2, current_to_rand1]
    mutations_names = [ x.__doc__ for x in mutations]
    n_operators = len(mutations)
    
    NP = int(NP)
    opu = np.full(NP, -1)
    
    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim, x_min, f_min = len(lbounds), (lbounds + ubounds) / 2, None
    X = lbounds + (ubounds - lbounds) * np.random.rand(NP, dim)
    X[0, :] = x0
    u = np.full((NP,dim), 0)

    F = np.apply_along_axis(fun, 1, X)
    #F = [fun(x) for x in X]
    max_budget = budget
    budget -= NP
    
    generation = 0
    best = np.argmin(F);
    # MANUEL: f_min is None!
    if f_min is None or F[best] < f_min:
        x_min, f_min = X[best, :], F[best];
    best_so_far = f_min
    best_so_far1 = best_so_far


    if mutation == "aos":
        aos_method = aos.Unknown_AOS(NP, n_ops = n_operators, OM_choice = OM_choice,
                                     rew_choice = rew_choice, rew_args = rew_args,
                                     qual_choice = qual_choice, qual_args = qual_args,
                                     prob_choice = prob_choice, prob_args = prob_args,
                                     select_choice = select_choice)
        select_mutation = aos_method.select_operator
    elif mutation == "random":
        select_mutation = lambda : np.random.randint(n_operators) 
    elif mutation in mutations_names:
        mutation = mutations_names.index(mutation)
        select_mutation = lambda : mutation
    else:
        raise ValueError("unknown mutation " + mutation)
    
    
    stats_file = None
    if stats_filename:
        stats_file = open(stats_filename, 'w+')
    #problem_data = "i" + "-" + str(uuid.uuid4()) + "-" + str(instance) + ".txt"

    trace = TraceFile(trace_filename, dim = dim, optimum = instance_best_value)
    trace.print(fun.evaluations, best_so_far, header = True)
    
    while budget > 0:
        
        fill_points = np.random.randint(dim, size = NP)
        
        for i in range(NP):
            # No mutation strategy needs more than 5.
            r = select_samples(NP, i, 5)
            best = np.argmin(F)
            crossovers = (np.random.rand(dim) < CR)
            crossovers[fill_points[i]] = True
            # MANUEL: What is this trial?
            # trial = aos_method.X[i]
            SI = select_mutation()
            # if stats_file:
            #     stats_file.write("{} {}\n".format(generation, SI))
            assert SI >= 0 and SI <= len(mutations)
            opu[i] = SI
            mutate = mutations[SI]
            bprime = mutate(X, r, best, FF)
            u[i,:] = np.where(crossovers, bprime, X[i, :])
    
        F1 = np.apply_along_axis(fun, 1, u)

        if mutation == "aos":
            aos_method.OM_Update(F, F1, F_bsf = best_so_far, opu = opu)
                
        #output_file.write(str(aos_method.reward)+"\n")
        #output_file.write(str(aos_method.quality)+"\n")
        #output_file.write(str(aos_method.probability)+"\n")
        #fitness_swap = [a<p for a,p in zip(F1,F)]
        F = np.where(F1 <= F, F1, F)
        X[F1 <= F, :] = u[F1 <= F, :]
        
        best = np.argmin(F)
        if f_min is None or F[best] < f_min:
            x_min, f_min = X[best, :], F[best]
            best_so_far1 = f_min;
        if best_so_far1 < best_so_far:
            best_so_far = best_so_far1
            trace.print(fun.evaluations, best_so_far)

        generation += 1
        budget -= NP

    if mutation == "aos":
        aos_method.gen_window.write_to(sys.stderr)
    
    #output_file.write("Last generation number"+str(generation)+".................................................................................\n")
    #output_file.close()
    #print("one configuartion tested on one instance")
    #file.close()
    return best_so_far

