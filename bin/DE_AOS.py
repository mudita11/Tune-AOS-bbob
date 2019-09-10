#!/usr/bin/env python3
"""Python script for the COCO experimentation module `cocoex`.

Usage from a system shell::

    python DE_AOS.py bbob

runs a full but short experiment on the bbob suite. The optimization
algorithm used is determined by the `SOLVER` attribute in this file::

    python DE_AOS.py bbob 20

runs the same experiment but with a budget of 20 * dimension
f-evaluations::

    python DE_AOS.py bbob-biobj 1e3 1 20

runs the first of 20 batches with maximal budget of
1000 * dimension f-evaluations on the bbob-biobj suite.
All batches must be run to generate a complete data set.

Usage from a python shell:

>>> import DE_AOS as ee
>>> ee.suite_name = "bbob-biobj"
>>> ee.SOLVER = ee.random_search  # which is default anyway
>>> ee.observer_options['algorithm_info'] = "default of DE_AOS.py"
>>> ee.main(5, 1+9, 2, 300)  # doctest: +ELLIPSIS
Benchmarking solver...

runs the 2nd of 300 batches with budget 5 * dimension and at most 9 restarts.

Calling `DE_AOS.py` without parameters prints this
help and the available suite names.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
try: range = xrange
except NameError: pass
import os, sys
import time
import numpy as np  # "pip install numpy" installs numpy
from cocoex import Suite, Observer, log_level, known_suite_names
del absolute_import, division, print_function, unicode_literals
import random
import math
import csv
from numpy.linalg import inv
import shutil

import aos
import de, R_de

verbose = 1

try: import cma  # cma.fmin is a solver option, "pip install cma" installs cma
except: pass
try: from scipy.optimize import fmin_slsqp  # "pip install scipy" installs scipy
except: pass
try: range = xrange  # let range always be an iterator
except NameError: pass

def debug_print(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def default_observers(update={}):
    """return a map from suite names to default observer names"""
    # this is a function only to make the doc available and
    # because @property doesn't work on module level
    _default_observers.update(update)
    return _default_observers
_default_observers = {
    'bbob': 'bbob',
    'bbob-biobj': 'bbob-biobj',
    'bbob-biobj-ext': 'bbob-biobj',
    'bbob-constrained': 'bbob',
    'bbob-largescale': 'bbob',  # todo: needs to be confirmed
    }
def default_observer_options(budget_=None, suite_name_=None):
    """return defaults computed from input parameters or current global vars
    """
    global budget, suite_name, result_folder
    if budget_ is None:
        budget_ = budget
    if suite_name_ is None:
        suite_name_ = suite_name
    opts = {}
    try:
        if result_folder is not None and result_folder != "":
            opts.update({'result_folder': '%s-%s_on_%s_budget%04dxD'
                         % (result_folder, SOLVER.__name__, suite_name_, budget_)})
        else:
            opts.update({'result_folder': '%s-%s_on_%s_budget%04dxD'
                         % (result_folder, SOLVER.__name__, suite_name_, budget_)})
    except: pass
    try:
        solver_module = '(%s)' % SOLVER.__module__
    except:
        solver_module = ''
    try:
        opts.update({'algorithm_name': SOLVER.__name__ + solver_module})
    except: pass
    return opts

class ObserverOptions(dict):
    """a `dict` with observer options which can be passed to
    the (C-based) `Observer` via the `as_string` property.
    
    See http://numbbo.github.io/coco-doc/C/#observer-parameters
    for details on the available (C-based) options.

    Details: When the `Observer` class in future accepts a dictionary
    also, this class becomes superfluous and could be replaced by a method
    `default_observer_options` similar to `default_observers`.
    """
    def __init__(self, options={}):
        """set default options from global variables and input ``options``.

        Default values are created "dynamically" based on the setting
        of module-wide variables `SOLVER`, `suite_name`, and `budget`.
        """
        dict.__init__(self, options)
    def update(self, *args, **kwargs):
        """add or update options"""
        dict.update(self, *args, **kwargs)
        return self
    def update_gracefully(self, options):
        """update from each entry of parameter ``options: dict`` but only
        if key is not already present
        """
        for key in options:
            if key not in self:
                self[key] = options[key]
        return self
    @property
    def as_string(self):
        """string representation which is accepted by `Observer` class,
        which calls the underlying C interface
        """
        s = str(self).replace(',', ' ')
        for c in ["u'", 'u"', "'", '"', "{", "}"]:
            s = s.replace(c, '')
        return s

def print_flush(*args):
    """print without newline but with flush"""
    print(*args, end="")
    sys.stdout.flush()


def ascetime(sec):
    """return elapsed time as str.

    Example: return `"0h33:21"` if `sec == 33*60 + 21`. 
    """
    h = sec / 60**2
    m = 60 * (h - h // 1)
    s = 60 * (m - m // 1)
    return "%dh%02d:%02d" % (h, m, s)


class ShortInfo(object):
    """print minimal info during benchmarking.

    After initialization, to be called right before the solver is called with
    the respective problem. Prints nothing if only the instance id changed.

    Example output:

        Jan20 18h27:56, d=2, running: f01f02f03f04f05f06f07f08f09f10f11f12f13f14f15f16f17f18f19f20f21f22f23f24f25f26f27f28f29f30f31f32f33f34f35f36f37f38f39f40f41f42f43f44f45f46f47f48f49f50f51f52f53f54f55 done

        Jan20 18h27:56, d=3, running: f01f02f03f04f05f06f07f08f09f10f11f12f13f14f15f16f17f18f19f20f21f22f23f24f25f26f27f28f29f30f31f32f33f34f35f36f37f38f39f40f41f42f43f44f45f46f47f48f49f50f51f52f53f54f55 done

        Jan20 18h27:57, d=5, running: f01f02f03f04f05f06f07f08f09f10f11f12f13f14f15f16f17f18f19f20f21f22f23f24f25f26f27f28f29f30f31f32f33f34f35f36f37f38f39f40f41f42f43f44f45f46f47f48f49f50f51f52f53f54f55 done

    """
    def __init__(self):
        self.f_current = None  # function id (not problem id)
        self.d_current = 0  # dimension
        self.t0_dimension = time.time()
        self.evals_dimension = 0
        self.evals_by_dimension = {}
        self.runs_function = 0
    def print(self, problem, end="", **kwargs):
        print(self(problem), end=end, **kwargs)
        sys.stdout.flush()
    def add_evals(self, evals, runs):
        self.evals_dimension += evals
        self.runs_function += runs
    def dimension_done(self):
        self.evals_by_dimension[self.d_current] = (time.time() - self.t0_dimension) / self.evals_dimension
        s = '\n    done in %.1e seconds/evaluation' % (self.evals_by_dimension[self.d_current])
        # print(self.evals_dimension)
        self.evals_dimension = 0
        self.t0_dimension = time.time()
        return s
    def function_done(self):
        s = "(%d)" % self.runs_function + (2 - int(np.log10(self.runs_function))) * ' '
        self.runs_function = 0
        return s
    def __call__(self, problem):
        """uses `problem.id` and `problem.dimension` to decide what to print.
        """
        f = "f" + problem.id.lower().split('_f')[1].split('_')[0]
        res = ""
        if self.f_current and f != self.f_current:
            res += self.function_done() + ' '
        if problem.dimension != self.d_current:
            res += '%s%s, d=%d, running: ' % (self.dimension_done() + "\n\n" if self.d_current else '',
                        ShortInfo.short_time_stap(), problem.dimension)
            self.d_current = problem.dimension
        if f != self.f_current:
            res += '%s' % f
            self.f_current = f
        # print_flush(res)
        return res
    def print_timings(self):
        print("  dimension seconds/evaluations")
        print("  -----------------------------")
        for dim in sorted(self.evals_by_dimension):
            print("    %3d      %.1e " %
                  (dim, self.evals_by_dimension[dim]))
        print("  -----------------------------")
    @staticmethod
    def short_time_stap():
        l = time.asctime().split()
        d = l[0]
        d = l[1] + l[2]
        h, m, s = l[3].split(':')
        return d + ' ' + h + 'h' + m + ':' + s

def EA_AOS(fun, x0, lbounds, ubounds, budget, instance):
    '''instance is the problem_index (an integer).
    
    There are five classes in bbob each consisting of functions with different properties. For instance a fun, f1, from first class is sphere function. Now there are 15 instances of each fun. Eg. for f1 translated or shifted versions are instances.
    Problem is represented as (f_i, n, j, t): f_i is i-fun, n is dimension, j is instance number and t is target.'''

    ## WARNING: These numbers are taken from the files generated by BBOB. This
    ## is probably only valid for bbob with dimension: 20
    # FIXME: This should be read by the target-runner from coco or moved to a
    # text file that the target-runner can read.
    assert suite_options == "dimensions: 20" and suite_name == "bbob"
    opt = {4:-2.525000000000e+01,
           15: -2.098800000000e+02,
           27: -5.688800000000e+02,
           30: -4.620900000000e+02,
           44: 4.066000000000e+01,
           50: -3.930000000000e+01,
           65: -6.639000000000e+01,
           70: 9.953000000000e+01,
           81: 3.085000000000e+01,
           90: 9.294000000000e+01,
           92: 3.820000000000e+00,
           111: -1.897900000000e+02,
           120: 1.238300000000e+02,
           130: -4.840000000000e+00,
           140: -5.191000000000e+01,
           158: -2.033000000000e+01,
           179: 7.789000000000e+01,
           188: -2.229800000000e+02,
           200: 3.270000000000e+01,
           201: -3.943000000000e+01,
           203: 7.640000000000e+00,
           209: -9.925000000000e+01,
           217: -3.475000000000e+01,
           219: -9.247000000000e+01,
           244: -1.479000000000e+02,
           250: 4.739000000000e+01,
           255: -1.694000000000e+01,
           257: 2.731500000000e+02,
           277: -2.602000000000e+01,
           281: -1.035000000000e+01,
           290: -1.367600000000e+02,
           299: -1.455800000000e+02,
           311: -4.860000000000e+01,
           321: 9.980000000000e+01,
           333: -2.231200000000e+02,
           349: -1.335900000000e+02}

    instance_best_value = 0
    if instance in opt:
        instance_best_value = opt[instance]
    
    cost = de.DE(fun, x0, lbounds, ubounds, budget, instance, instance_best_value,
                 trace_filename, stats_filename,
                 # DE parameters
                 FF, CR, NP, top_NP, mutation = mutation,
                 # Offspring Metrics
                 OM_choice = OM_choice,
                 # Rewards
                 rew_choice = rew_choice, rew_args = rew_args,
                 # Qualities
                 qual_choice = qual_choice, qual_args = qual_args,
                 # Probabilities
                 prob_choice = prob_choice, prob_args = prob_args,
                 # Selection
                 select_choice = select_choice, select_args = select_args)
    print("\n",cost)
    return cost


# ===============================================
# loops over a benchmark problem suite
# ===============================================
def batch_loop(solver, suite, observer, budget,
               max_runs, current_batch, number_of_batches, problem_subset):
    """loop over all problems in `suite` calling
    `coco_optimize(solver, problem, budget * problem.dimension, max_runs)`
    for each eligible problem.

    A problem is eligible if ``problem_index + current_batch - 1``
    modulo ``number_of_batches`` equals ``0``.

    This distribution into batches is likely to lead to similar
    runtimes for the batches, which is usually desirable.

    problem_subset is a list of problem_index(es). If empty, run all problems.
    """
    global train_or_test, cost_best
    addressed_problems = []
    short_info = ShortInfo()
    # problem_subset = range(165, 360)
    for problem_index, problem in enumerate(suite):
        if problem_subset and problem_index not in problem_subset:
            continue
        if (problem_index + current_batch - 1) % number_of_batches:
            continue
        observer.observe(problem)
        short_info.print(problem) if verbose else None
        runs = coco_optimize(solver, problem, budget * problem.dimension,
                             problem_index, max_runs)
        if verbose:
            print_flush("!" if runs > 2 else ":" if runs > 1 else ".")
        short_info.add_evals(problem.evaluations, runs)
        problem.free()
        addressed_problems += [problem.id]
    print(short_info.function_done() + short_info.dimension_done())
    short_info.print_timings()
    print("  %s done (%d of %d problems benchmarked%s)" %
           (suite_name, len(addressed_problems), len(suite),
             ((" in batch %d of %d" % (current_batch, number_of_batches))
               if number_of_batches > 1 else "")), end="")
    if number_of_batches > 1:
        print("\n    MAKE SURE TO RUN ALL BATCHES", end="")
    if train_or_test == "train" and cost_best == "yes":
        print("\nDeleting folder: ", observer.result_folder)
        shutil.rmtree(os.path.abspath(observer.result_folder), ignore_errors = True)
    
    return addressed_problems

#===============================================
# interface: ADD AN OPTIMIZER BELOW
#===============================================
def coco_optimize(solver, fun, max_evals, problem_index, max_runs=1):
    """`fun` is a callable, to be optimized by `solver`.

    The `solver` is called repeatedly with different initial solutions
    until either the `max_evals` are exhausted or `max_run` solver calls
    have been made or the `solver` has not called `fun` even once
    in the last run.

    Return number of (almost) independent runs.
    """
    # Receives max_evaulations as budget * dim

    range_ = fun.upper_bounds - fun.lower_bounds
    center = fun.lower_bounds + range_ / 2
    if fun.evaluations:
        print('WARNING: %d evaluations were done before the first solver call' %
              fun.evaluations)

    for restarts in range(int(max_runs)):
        remaining_evals = max_evals - fun.evaluations
        x0 = center + (restarts > 0) * 0.8 * range_ * (
                np.random.rand(fun.dimension) - 0.5)
        #fun(x0)  # can be incommented, if this is done by the solver

        if solver.__name__ in ("random_search", ):
            solver(fun, fun.lower_bounds, fun.upper_bounds,
                   remaining_evals)
        elif solver.__name__ == 'fmin' and solver.__globals__['__name__'] in ['cma', 'cma.evolution_strategy', 'cma.es']:
            if x0[0] == center[0]:
                sigma0 = 0.02
                restarts_ = 0
            else:
                x0 = "%f + %f * np.random.rand(%d)" % (
                        center[0], 0.8 * range_[0], fun.dimension)
                sigma0 = 0.2
                restarts_ = 6 * (observer_options.as_string.find('IPOP') >= 0)

            solver(fun, x0, sigma0 * range_[0], restarts=restarts_,
                   options=dict(scaling=range_/range_[0], maxfevals=remaining_evals,
                                termination_callback=lambda es: fun.final_target_hit,
                                verb_log=0, verb_disp=0, verbose=-9))
        elif solver.__name__ == 'fmin_slsqp':
            solver(fun, x0, iter=1 + remaining_evals / fun.dimension,
                   iprint=-1)
############################ ADD HERE ########################################
        # ### IMPLEMENT HERE THE CALL TO ANOTHER SOLVER/OPTIMIZER ###
        # elif True:
        #     CALL MY SOLVER, interfaces vary
##############################################################################
        elif True:
            solver(fun, x0, fun.lower_bounds, fun.upper_bounds, remaining_evals, problem_index)
        else:
            raise ValueError("no entry for solver %s" % str(solver.__name__))
        if fun.evaluations >= max_evals or fun.final_target_hit:
            break
        # quit if fun.evaluations did not increase
        #print("fun.final_target_hit: ",fun.final_target_hit);
        if fun.evaluations <= max_evals - remaining_evals:
            if max_evals - fun.evaluations > fun.dimension + 1:
                print("WARNING: %d evaluations remaining" %
                      remaining_evals)
            if fun.evaluations < max_evals - remaining_evals:
                raise RuntimeError("function evaluations decreased")
            break
    return restarts + 1

# ===============================================
# set up: CHANGE HERE SOLVER AND FURTHER SETTINGS AS DESIRED
# ===============================================
######################### CHANGE HERE ########################################
# CAVEAT: this might be modified from input args
suite_name = "bbob"  # always overwritten when called from system shell
                     # see available choices via cocoex.known_suite_names
max_runs = 1  # number of (almost) independent trials per problem instance
number_of_batches = 1  # allows to run everything in several batches
current_batch = 1      # 1..number_of_batches
##############################################################################
#SOLVER = random_search
# SOLVER = my_solver # SOLVER = fmin_slsqp # SOLVER = cma.fmin
SOLVER = EA_AOS
suite_instance = "" # "year:2016"
suite_options = "dimensions: 20"   #"dimensions: 2,3,5,10,20 "  # if 40 is not desired
# for more suite options, see http://numbbo.github.io/coco-doc/C/#suite-parameters
observer_options = ObserverOptions({  # is (inherited from) a dictionary
                    #'algorithm_info': "AN AOS ALGORITHM", # CHANGE/INCOMMENT THIS!
                    'algorithm_info': "Generic AOS framework", # CHANGE/INCOMMENT THIS!
                    # 'algorithm_name': "",  # default already provided from SOLVER name
                    # 'result_folder': "",  # default already provided from several global vars
                   })
######################### END CHANGE HERE ####################################

# ===============================================
# run (main)
# ===============================================
def main(problem_subset, budget,
         max_runs=max_runs,
         current_batch=current_batch,
         number_of_batches=number_of_batches):
    """Initialize suite and observer, then benchmark solver by calling
    ``batch_loop(SOLVER, suite, observer, budget,...``
    """
    observer_name = default_observers()[suite_name]
    observer_options.update_gracefully(default_observer_options())

    observer = Observer(observer_name, observer_options.as_string)
    suite = Suite(suite_name, suite_instance, suite_options)

    print("Benchmarking solver '%s' with budget=%d*dimension on %s suite, %s"
          % (' '.join(str(SOLVER).split()[:2]), budget,
             suite.name, time.asctime()))
    if number_of_batches > 1:
        print('Batch usecase, make sure you run *all* %d batches.\n' %
              number_of_batches)
    t0 = time.clock()
    batch_loop(SOLVER, suite, observer, budget, max_runs,
               current_batch, number_of_batches,
               problem_subset=problem_subset)
    print(", %s (%s total elapsed time)." % 
            (time.asctime(), ascetime(time.clock() - t0)))

# ===============================================
from argparse import ArgumentParser,RawDescriptionHelpFormatter,_StoreTrueAction,ArgumentDefaultsHelpFormatter

if __name__ == '__main__':
    """read input parameters and call `main()`"""

    description = __doc__ + "\n" + "Recognized suite names: " + str(known_suite_names)

    
    class RawDesArgDefaultsHelpFormatter(ArgumentDefaultsHelpFormatter,
                                               RawDescriptionHelpFormatter):
        pass

    parser = ArgumentParser(description = description,
                            formatter_class=RawDesArgDefaultsHelpFormatter)
    parser.add_argument('suite_name', help='suite name, e.g., bbob', choices = known_suite_names)
    parser.add_argument('budget', metavar='budget', type=int, help='function evaluations = BUDGET * dimension')
    parser.add_argument('current_batch', type=int, default=1, help='batch to run')
    parser.add_argument('number_of_batches', type=int, default=1, help='number of batches')
    # nargs= is the correct way to handle accepting multiple arguments.
    # '+' == 1 or more.
    # '*' == 0 or more.
    # '?' == 0 or 1.
    parser.add_argument('-i', '--instance', nargs='+', type=int, default=[], help='problem instance to train on (multiple numbers are possible)')
    parser.add_argument('--seed', type=int, default=0, help='seed to initialise population')
    parser.add_argument('--trace', help='file to store fevals fitness progress')
    parser.add_argument('--stats', help='file to store statistics about the evolution')
    parser.add_argument('--result_folder', default="", help='file to store statistics about the evolution')
    parser.add_argument('--train_test', default="train", help = 'train or test option')
    parser.add_argument('--cost_best', default="yes", help = 'cost is minimising best or not')
    
    class dump_irace_parameters(_StoreTrueAction):
        def __call__(self, parser, namespace, values, option_string=None):
            print(de.DE_irace_parameters())
            print(aos.Unknown_AOS.irace_parameters())
            print(aos.ProbabilityType.irace_parameters())
            print(aos.RewardType.irace_parameters())
            print(aos.QualityType.irace_parameters())
            print(aos.SelectionType.irace_parameters())
            debug_print("Dumping irace parameter file of known AOS:")
            for key in aos.Unknown_AOS.known_AOS.keys():
                filename = key + ".txt"
                if os.path.isfile(filename):
                    print("error: File " + filename + " already exists!")
                    parser.exit(1)
                with open(filename, "w") as f:
                    debug_print("Creating", filename)
                    output = "##### AOS:  " + key + ".txt\n"
                    f.write(de.DE_irace_parameters(override = dict(mutation=["known_aos"])))
                    f.write(aos.Unknown_AOS.irace_dump_knownAOS(key))

            parser.exit(0)
        
    parser.add_argument('--irace', action=dump_irace_parameters, help='dump parameters.txt for irace')

    # DE parameters
    de.DE_add_arguments(parser)
    # Handle Offspring metric
    aos.Unknown_AOS.add_argument(parser)
    # Handle rewards
    # This uses __subclasses__ to find choices.
    rew_args_names = aos.RewardType.add_argument(parser)
    # Handle qualities
    qual_args_names = aos.QualityType.add_argument(parser)
    # Handle probabilities
    prob_args_names = aos.ProbabilityType.add_argument(parser)
    # Handle Selection
    select_args_names = aos.SelectionType.add_argument(parser)
    
    args = parser.parse_args()

    suite_name = args.suite_name
    budget = args.budget
    current_batch = args.current_batch
    number_of_batches = args.number_of_batches
    result_folder = args.result_folder
    train_or_test = args.train_test
    cost_best = args.cost_best
    
    instance =  args.instance
    trace_filename = args.trace
    stats_filename = args.stats
    
    ## FIXME: At some moment we could replace explicit parsing by implicit DE(**de_args)
    # de_args = dict.fromkeys(de_args, None)
    # for x in de_args.keys():
    #     de_args[x] = getattr(args, x)
    FF = args.FF
    CR = args.CR
    NP = args.NP
    top_NP = args.top_NP
    mutation = args.mutation
    budget = args.budget
    
    seed = args.seed
    # If no seed is given, we generate one.
    if seed == 0:
        seed = np.random.randint(0, 2**32 - 1, 1)
    np.random.seed(seed)

    # Handle Offpring Metrics
    OM_choice = args.OM_choice
    
    # Handle rewards
    rew_choice = args.rew_choice
    rew_args = {}
    for x in rew_args_names:
        rew_args[x] = getattr(args, x)
    
    # Handle qualities
    qual_choice = args.qual_choice
    qual_args = {}
    for x in qual_args_names:
        qual_args[x] = getattr(args, x)

    # Handle probabilities
    prob_choice = args.prob_choice
    prob_args = {}
    for x in prob_args_names:
        prob_args[x] = getattr(args, x)

    # Handle selection
    select_choice = args.select_choice
    select_args = {}
    for x in select_args_names:
        select_args[x] = getattr(args, x)
    
    # FIXME: instance should be the last argument to match how other functions
    # work.
    main(instance, budget, max_runs, current_batch, number_of_batches)

