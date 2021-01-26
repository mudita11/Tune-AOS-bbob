__version__ = '1.0'
# FIXME: Find all non-ascii characters and replace them for their ASCII equivalent.
import sys
import copy
import math
import numpy as np
from scipy.stats import rankdata
from scipy.spatial import distance
from collections import deque

import warnings
warnings.filterwarnings('error', "divide by zero encountered in true_divide")
#np.seterr(all='raise')

from abc import ABC,abstractmethod

# A small epsilon number to avoid division by 0.
EPSILON = np.finfo(np.float32).eps

import gzip
class DebugFile():
    def __init__(self, filename, suffix, header, n_ops):
        self._fh = None
        if filename:
            filename += suffix + ".gz"
            print(f"Writing to file {filename}")
            self._file = gzip.open(filename, "wt")
            self._file.write("generation " + " ".join([header + str(i) for i in range(n_ops)]) + "\n")
            
    def write(self, generation, vector):
        if not self._file: return
        self._file.write(str(generation) + " ")
        np.savetxt(self._file, vector[None,:], fmt = "%10.8g")

    def __del__(self):
        if self._file:
            self._file.close()
            self._file = None

def all_subclasses(cls):
    return list(set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)]))

def debug_print(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def normalize_sum(x):
    # The divide by zero check does not work if we have negative values because
    # [5, -5] sums to zero.
    assert np.all(x >= 0.0)
    s = x.sum()
    # FIXME: use tolerance check
    if s != 0: return x / s
    x[:] = 1.0 / len(x)
    return x

def normalize_max(x):
    # The divide by zero check does not work if we have negative values.
    assert np.all(x >= 0.0)
    s = x.max()
    # FIXME: use tolerance check
    if s != 0: return x / s
    return x

def ucb1(n, C, p):
    '''Calculates Upper Confidence Bound (1)'''
    ucb = p + C * np.sqrt(2 * np.log(n.sum()) / n)
    #ucb[np.isinf(ucb) | np.isnan(ucb)] = 0.0
    return ucb

def get_choices_dict(cls):
    return { x.__name__:x for x in all_subclasses(cls)}

def get_choices(cls, override = []):
    """Get all possible choices of a component of AOS framework"""
    if len(override):
        choices = override
        if type(choices) == str:
            choices = [choices]
    else:
        choices = [x.__name__ for x in all_subclasses(cls)]
    choices_help = ', '.join(f"{i}" for i in choices)
    return choices, choices_help

def parser_add_arguments(cls, parser):
    "Helper function to add arguments of a class to an ArgumentParser"
    choices, choices_help = get_choices(cls)
    group = parser.add_argument_group(title=cls.__name__)
    group.add_argument("--"  + cls.param_choice, choices=choices,
                       help=cls.param_choice_help + " (" + choices_help + ")")
    names = []
    for i in range(0, len(cls.params), 5):
        arg, type, default, domain, help = cls.params[i:i+5]
        if type is None: continue
        if type == object: type = str
        #group.add_argument('--' + arg, type=type, default=default, help=help)
        group.add_argument('--' + arg, type=type, default=None, help=help)
        names.append(arg)
    # Return the names
    return names

def aos_irace_parameters(cls, override = {}):
    """All AOS components may call this function.
    override is an optional dictionary that allows to override some parameter values"""
    output = "# " + cls.__name__ + "\n"
    if cls.param_choice in override:
        choices, choices_help = get_choices(cls, override = override[cls.param_choice])
    else:
        choices, choices_help = get_choices(cls)

    output += irace_parameter(name=cls.param_choice, type=object, domain=choices, help=choices_help)
    for i in range(0, len(cls.params), 5):
        arg, type, default, domain, help = cls.params[i:i+5]
        if type is None: continue
        condition = irace_condition(cls.param_choice, cls.params_conditions[arg], override)
        if not condition is None:
            output += irace_parameter(name=arg, type=type, domain=domain,
                                      condition=condition, help=help, override=override)
    return output


def irace_parameter(name, type, domain, condition="", help="", override = {}):
    """Return a string representation of an irace parameter"""
    irace_types = {int:"i", float:"r", object: "c"}
    if name in override:
        domain = override[name]
        if isinstance(domain, tuple):
            domain = domain[0]
        
    arg = f'"--{name} "'
    if not hasattr(domain, '__len__') or isinstance(domain, str):
        domain = [domain]
    if len(domain) == 1:
        type = object
    domain = "(" + ", ".join([str(x) for x in domain]) + ")"
    if condition != "":
        condition = "| " + condition
    if help != "":
        help = "# " + help
    return f'{name:20} {arg:25} {irace_types[type]} {domain:20} {condition:30} {help}\n'

def as_r_string(value):
    if isinstance(value, str):
        return '"' + value + '"'
    else:
        return str(value)
    
def irace_condition(what, values, override = {}):
    """Return a string representation of the condition of an irace parameter"""
    if not values:
        return ""
    if what in override:
        values = [value for value in override[what] if value in values]
        if not values:
            return None
        
    if not hasattr(values, '__len__') or isinstance(values, str):
        values = [values]
    if len(values) == 1:
        return what + " == " + as_r_string(values[0])
    return what + " %in% c(" + ", ".join([as_r_string(x) for x in values]) + ")"

class GenWindow(object):
    ''''Generational Window of OM values. g=0 is the oldest, while g=len(self)-1 is the most recent generation'''
# FIXME (needs updating): gen_window stores the offspring metric data for each offspring when offspring is better than parent. Otherwise it stores np.nan for that offspring. Its a list. Its structre is as follows: [[[second_dim], [second_dim], [second_dim]], [[],[],[]], ...]. Second_dim represnts Offspring metric data for an offspring. The number of second dims will be equal to the population size, contained in third_dim. Third_dim represents a generation. Thus, [[],[],[]] has data of all offsprings in a generation."""
    
    def __init__(self, n_ops, max_gen = 0):
        self.n_ops = n_ops
        self.max_gen = max_gen
        # Private
        # A matrix of at most max_gen rows and with NP columns. Each entry
        # _gen_window_op[i,j] is the operator that created children j at
        # generation i.
        self._gen_window_op = None
        # A cubic matrix of at most max_gen rows and with len(metrics)
        # columns. Each entry _gen_window_met[i,j,k] gives the OM value of the
        # children j at generation i for metric k.
        self._gen_window_met = None

    def __len__(self):
        return self._gen_window_op.shape[0]

    def get_max_gen(self):
        return np.minimum(self.max_gen, len(self))

    def append(self, window_op, window_met):
        if self._gen_window_op is None:
            self._gen_window_op = np.array([window_op])
            self._gen_window_met = np.array([window_met])
        else:
            self._gen_window_op = np.append(self._gen_window_op, [window_op], axis = 0)
            self._gen_window_met = np.append(self._gen_window_met, [window_met], axis = 0)

    def count_succ_total(self, gen = None):
        """Counts the number of successful and total applications for each operator in generation 'gen'"""
        if gen is None:
            gen = slice(-self.max_gen, None, None)
        window_met = self._gen_window_met[gen, :].ravel()
        window_op = self._gen_window_op[gen, :].ravel()
        is_succ = ~np.isnan(window_met)
        total_success = np.zeros(self.n_ops, dtype=int)
        total_apps = np.zeros(self.n_ops, dtype=int)
        for op in range(self.n_ops):
            is_op = window_op == op
            total_success[op] = np.sum(is_op & is_succ)
            total_apps[op] = np.sum(is_op)
        # To avoid division by zero (assumes that value == 0 for these)
        total_apps[total_apps == 0] = 1
        return total_success, total_apps

    def _apply(self, function, gen = None):
        """Apply function to metric values at generation gen for all operators"""
        if gen is None:
            gen = slice(-self.max_gen, None, None)
        window_met = self._gen_window_met[gen, :].ravel()
        window_op = self._gen_window_op[gen, :].ravel()
        # Assign 0.0 to any entry that is nan
        is_not_succ = np.isnan(window_met)
        window_met[is_not_succ] = 0
        value = np.zeros(self.n_ops)
        for op in range(self.n_ops):
            is_op = window_op == op
            if is_op.any():
                value[op] = function(window_met[is_op])
        return value

    def sum_per_op(self, gen = None):
        """Return metric sum per op for each of the last max_gen generations"""
        return self._apply(np.sum, gen = gen)

    def max_per_op(self, gen = None):
        """Return metric sum per op for each of the last max_gen generations"""
        return self._apply(np.max, gen = gen)
    
    def is_success(self, gen):
        window_met = self._gen_window_met[gen, :]
        return ~np.isnan(window_met)

    def get_ops_of_child(self, i):
        ## FIXME: This implementation assumes that all parents of i are stored
        ## at the same place of the population, which is true for DE but
        ## not in general.
        return self._gen_window_op[-self.max_gen:, i]
        
    def write_to(self, filename):
        ops = self._gen_window_op
        met = self._gen_window_met
        gen = np.tile(np.arange(ops.shape[0]).reshape(-1,1), (1, ops.shape[1]))
        out = np.hstack((ops.reshape(-1,1),
                         gen.reshape(-1,1),
                         met.reshape(-1,1)))
        np.savetxt(filename, out, fmt= 2*["%d"] + 1*["%+20.15e"],
                   header = "operator generation metric")

class OpWindow(object):
    """ This window stores a sliding window of size maxsize per operator"""
    def __init__(self, n_ops, max_size = 0):
        self.max_size = max_size
        self.n_ops = n_ops
        if self.max_size > 0:
            # Matrix of metric values per op
            self.resize(max_size)

    def resize(self, max_size):
        self.max_size = max_size
        self._window = np.full((self.n_ops, self.max_size), np.inf)
        
    def append(self, ops, values):
        '''Push data of improved offspring in the window. It follows First In First Out Rule.'''
        assert len(ops) == len(values)
        if self.max_size == 0:
            return
        # We store only successes.
        is_success = ~np.isnan(values) 
        ops = ops[is_success]
        values = values[is_success]
        if len(values) > self.max_size:
            ops = ops[-self.max_size:]
            values = values[-self.max_size:]

        for op in np.unique(ops):
            # Shift contents FIFO
            len_values = (ops == op).sum()
            self._window[op, :] = np.roll(self._window[op, :], -len_values)
            # Overwrite
            self._window[op, -len_values:] = values[ops == op]
            
        #np.savetxt(sys.stderr, self._window, fmt="%8g")
        
class AppWindow(object):
    """ This window stores a single sliding window of size maxsize for all operators (window of all operator applications)"""
    def __init__(self, n_ops, max_size = 0):
        self.max_size = max_size
        self.n_ops = n_ops
        if self.max_size > 0:
            # Vector of operators
            self._window_op = np.full(max_size, -1)
            # Vector of metrics
            # np.inf means not initialized
            # np.nan means unsuccessful application
            self._window_met = np.full(max_size, np.inf)
                
    def resize(self, max_size):
        self.max_size = max_size
        self._window_op = np.full(max_size, -1)
        self._window_met = np.full(max_size, np.inf)
    
    def append(self, ops, values):
        '''Push data of improved offspring in the window. It follows First In First Out Rule.'''
        assert len(ops) == len(values)
        if self.max_size == 0:
            return
        # We store only successes.
        is_success = ~np.isnan(values) 
        ops = ops[is_success]
        values = values[is_success]
        if len(values) > self.max_size:
            ops = ops[-self.max_size:]
            values = values[-self.max_size:]

        len_values = len(values)
        # Shift contents FIFO
        self._window_op = np.roll(self._window_op, -len_values)
        self._window_met = np.roll(self._window_met, -len_values)
        # Overwrite
        self._window_op[-len_values:] = ops
        self._window_met[-len_values:] = values
        #np.savetxt(sys.stderr, [self._window_op, self._window_met], fmt="%8g")

    def count_ops(self):
        n = np.zeros(self.n_ops, dtype=int)
        op, count = np.unique(self._window_op, return_counts=True)
        n[op] = count
        return n

    def apply_per_op(self, fun):
        """Apply function to metric values per operator"""
        window_met = self._window_met
        window_op = self._window_op
        value = np.zeros(self.n_ops)
        for op in np.unique(self._window_op):
            value[op] = function(self._window_met[self._window_op == op])
        return value

    def sum_per_op(self):
        """Sum of metric values per operator"""
        return self.apply_per_op(np.sum)

    def max_per_op(self):
        """Sum of metric values per operator"""
        return self.apply_per_op(np.max)

    def mean_per_op(self):
        """Sum of metric values per operator"""
        return self.apply_per_op(np.mean)


def calc_diversity(parents, children):
    # Calculate average euclidean distance of each child to all parents.
    return distance.cdist(children, parents).sum(axis=1)

class Metrics(object):

    # FIXME: can we get these names by listing the methods of the Metric class?
    OM_choices = {
        "absolute_fitness": 0, # "offsp_fitness"
        "exp_absolute_fitness": 1,
        "improv_wrt_parent": 2,
        "improv_wrt_pop": 3,
        "improv_wrt_bsf": 4,
        "improv_wrt_median": 5,
        "relative_fitness_improv": 6
    }
    param_choice = "OM_choice"
    param_choice_help = "Offspring metric selected"
    
    @classmethod
    def add_arguments(cls, parser):
        metrics_names = list(cls.OM_choices)
        parser.add_argument("--" + cls.param_choice, choices=metrics_names,
                            help=cls.param_choice_help)


    @classmethod
    def irace_parameters(cls, override = {}):
        metrics_names = list(cls.OM_choices)
        #choices = range(1, 1 + len(metrics_names))
        if cls.param_choice in override:
            #choices = override[cls.param_choice]
            metrics_names = override[cls.param_choice]
            #choices_help = ', '.join(f"{i}:{j}" for i,j in zip(choices, metrics_names))
        return irace_parameter(cls.param_choice, object, metrics_names,
                               help=cls.param_choice_help)

    def __init__(self, minimize, choice = None):
        # We have to define it here and not a class level, because we need 'self'
        OM_choices_fun = {
            "absolute_fitness": self.absolute_fitness,
            "exp_absolute_fitness": self.exp_absolute_fitness,
            "improv_wrt_parent": self.improv_wrt_parent,
            "improv_wrt_pop": self.improv_wrt_pop,
            "improv_wrt_bsf": self.improv_wrt_bsf,
            "improv_wrt_median": self.improv_wrt_median,
            "relative_fitness_improv": self.relative_fitness_improv
        }
        # FIXME: Extend to handle both minimization and maximization.
        assert minimize == True
        self.upper_bound = None
        self.choice = Metrics.OM_choices[choice]
        self.calc_metric = OM_choices_fun[choice]

    # Updates information and calculates the metric values.
    def update(self, F_children, F_parents, F_bsf):
        self.F_children = F_children
        self.F_parents = F_parents
        # Previous bsf
        self.F_bsf = F_bsf
        # New bsf
        self.F_new_bsf = min(F_bsf, F_children.min())
        # Best parent
        self.F_best = F_parents.min() 
        self.F_median = np.median(F_parents)

        return self.calc_metric()

    def filter_non_improved(self, values):
        # if child is worse or equal to parent, store np.nan.
        return np.where(self.F_children < self.F_parents, values, np.nan)
        
    # Fitness is minimised but metric is maximised
    def absolute_fitness(self):
        if self.upper_bound is None:
            # If all values are negative, then we need abs()
            self.upper_bound = max(abs(self.F_children.max()),abs(self.F_parents.max())) * 10
        # If a later individual is still worse than the upper bound, it is
        # probably very bad anyway.
        values = np.maximum(0, self.upper_bound - self.F_children)
        assert (values >= 0).all(),f'F_children = {self.F_children[values <= 0]}, U = {self.upper_bound}'
        values = self.filter_non_improved(values)
        return values

    def exp_absolute_fitness(self):
        values = softmax(-self.F_children)
        values = self.filter_non_improved(values)
        return values

    def improv_wrt_parent(self):
        values = self.F_parents - self.F_children
        values = np.where(values > 0, values, np.nan)
        return values

    def improv_wrt_pop(self):
        values = self.F_best - self.F_children
        values = np.where(values > 0, values, np.nan)
        return values
    
    def improv_wrt_bsf(self):
        values = self.F_bsf - self.F_children
        values = np.where(values > 0, values, np.nan)
        return values
    
    def improv_wrt_median(self):
        values = self.F_median - self.F_children
        values = np.where(values > 0, values, np.nan)
        return values
    
    def relative_fitness_improv(self):
        # MUDITA: if F_bsf = 1 and F1 = 0, then F_bsf / (F1 + eps) becomes 8388608.0 which is wrong.
        ## We need to use the new bsf because we want to use the lowest value
        ## ever and we want beta <= 1.
        assert np.all(self.F_new_bsf != 0), f"F_bsf = {self.F_new_bsf}"
        assert np.all(self.F_new_bsf != 0), f"F_bsf = {self.F_new_bsf}"
        # We use abs to ignore negative numbers. 
        beta = (abs(self.F_new_bsf) + EPSILON) / (np.abs(self.F_children) + EPSILON)
        # If we do have negative numbers, we may get values >= 1, just reverse.
        beta = np.where(beta > 1, 1 / beta, beta)
        assert ((beta > 0) & (beta <= 1)).all(),f"beta={beta}, F_bsf = {self.F_new_bsf}, F_children = {np.abs(self.F_children) + EPSILON}"
        return self.improv_wrt_parent() * beta



class AOS(object):
    
    known_AOS = {
        "COBRA" : {
            "OM_choice" : "improv_wrt_parent",
            "rew_choice" : "AvgMetric",
            "periodic" : True,
            "max_gen" : ([1,50], 10), # default 10
            "qual_choice" : "QualityIdentity",
            "prob_choice" : "LinearRank",
            "select_choice" : "ProportionalSelection",
        },
        # B. A. Julstrom, “What have you done for me lately? adapting operator probabilities in a steady-state genetic algorithm,” in ICGA, L. J. Eshelman, Ed. Morgan Kaufmann Publishers, San Francisco, CA, 1995, pp. 81–87.
        "ADOPP" : {
            "OM_choice" : "improv_wrt_median",
            "rew_choice" : "AncestorSuccess",
            "max_gen" : ([1, 50], 5), # default 5
            "decay": ([0.0, 1.0], 0.8), # default 0.8
            "app_winsize": ([1, 500], 100), # qlen = 100
            "periodic" : False,
            "frac": 0.0,
            "convergence_factor" : 0,
            "qual_choice" : "QualityIdentity",
            "prob_choice" : "ProbabilityMatching",
            "p_min" : 0.0,
            "select_choice" : "ProportionalSelection",
        },
        # B. A. Julstrom, “Adaptive operator probabilities in a genetic algorithm that applies three operators” in Proceedings of the 1997 ACM Symposium on Applied Computing, ser. SAC’97. New York, NY: ACM Press, 1997, pp. 233–238.
        "ADOPP_ext" : {
            "OM_choice" : "improv_wrt_median",
            "rew_choice" : "AncestorSuccess",
            "max_gen" : ([1, 50], 5), # default 5
            "decay": ([0.0, 1.0], 0.8), # default 0.8
            "app_winsize": ([1, 500], 100), # qlen = 100
            "periodic" : False,
            "frac": ([0.01, 0.1], 0.01), # default 0.01
            "convergence_factor" : ([1, 100], 20), # default 20
            "qual_choice" : "QualityIdentity",
            "prob_choice" : "ProbabilityMatching",
            "p_min" : 0.0,
            "select_choice" : "ProportionalSelection",
        },
        #  F. G. Lobo and D. E. Goldberg, “Decision making in a hybrid genetic
        #  algorithm,” in Proceedings of the 1997 IEEE International Conference on
        #  Evolutionary Computation (ICEC’97), T. Bäck, Z. Michalewicz, and X. Yao,
        #  Eds. Piscataway, NJ: IEEE Press, 1997, pp. 121–125.
        "Hybrid" : {
            "OM_choice" : "improv_wrt_pop",
            "rew_choice" : "ExtMetric", 
            "max_gen" : 1,
            "gamma" : 1,
            "periodic" : False,
            "qual_choice" : "Qlearning",
            "q_min" : 0.0, 
            "decay_rate" : ([0.001, 1.0], 0.1),
            "prob_choice" : "ProbabilityMatching",
            "p_min" : 0.0, # no default given.
            "select_choice" : "ProportionalSelection",
        },
        # C. Igel and M. Kreutz, “Using fitness distributions to improve the
        # evolution of learning structures,” in Proceedings of the 1999
        # Congress on Evolutionary Computation (CEC 1999), vol. 3. Piscataway,
        # NJ: IEEE Press, 1999, pp. 1902–1909.
        "Op_adapt" : {
            "OM_choice" : "improv_wrt_pop",
            "rew_choice" : "TotalGenAvg",
            "periodic" : 1,
            "max_gen" : ([1, 50], 10), # default 10
            "qual_choice" : "Qlearning",
            "decay_rate" : ([0.001, 1], 0.5), # default: 0.5 
            "q_min" : ([0.01, 1], 0.1), # default 0.1
            "prob_choice" : "ProbabilityMatching",
            "p_min" : 0.0,
            "select_choice" : "ProportionalSelection",
        },
        # J. T. Stanczak, J. J. Mulawka, and B. K. Verma, “Genetic algorithms with adaptive probabilities of operators selection,” in Proceedings Third International Conference on Computational Intelligence and Multimedia Applications. ICCIMA’99. IEEE, 1999, pp. 464—-468.
        "APOS" : {
            "OM_choice" : "improv_wrt_bsf",
            "rew_choice" : "NormAvgMetric",
            "periodic" : 0,
            "max_gen" : 1,
            "qual_choice" : "Qdecay",
            "decay_rate" : ([0.01, 1], 0.4),
            "q_min" : ([0.01, 1], 0.01),
            "prob_choice" : "ProbabilityMatching",
            "p_min" : 0.0,
            "select_choice" : "ProportionalSelection",
        },
        # J. Niehaus and W. Banzhaf, “Adaption of operator probabilities in
        # genetic programming,” in Proceedings of the 4th European Conference on
        # Genetic Programming, EuroGP 2001, ser. LNCS, J. Miller, M. Tomassini,
        # P. L. Lanzi, C. Ryan, A. G. B. Tettamanzi, and W. B. Langdon,
        # Eds. Springer, 2001, vol. 2038, pp. 325–336.
        "PDP" : {
            "OM_choice" : "improv_wrt_parent",
            "rew_choice" : "SuccessRate",
            "periodic": 0,
            "max_gen" : 100000, # default is np.inf
            "gamma": 2,
            "qual_choice" : "QualityIdentity",
            "prob_choice" : "ProbabilityMatching",
            "p_min" : ([0.0, 1.0], 0.2), # default is 0.2 / n_ops
            "select_choice" : "ProportionalSelection",
        },
        # C. Igel and M. Kreutz, “Operator adaptation in evolutionary computation
        # and its application to structure optimization of neural networks,”
        # Neurocomputing, vol. 55, no. 1-2, pp. 347–361, 2003.
        "Adapt_NN" : {
            "OM_choice" : "improv_wrt_parent",
            "rew_choice" : "NormAvgMetric",
            "periodic" : 1,
            "max_gen" : ([1, 50], 4), # default 4
            "qual_choice" : "Qlearning",
            "decay_rate": ([0.01, 1.0], 0.3), # delta default: 0.3
            "q_min" : 1, # actually 1 / n_ops
            "prob_choice" : "ProbabilityMatching",
            "p_min" : ([0.01, 1.0], 0.1), # default pmin = 0.1
            "select_choice" : "ProportionalSelection",
        },
        # Y. S. Ong and A. J. Keane, “Meta-Lamarckian learning in memetic algorithms,
        # ”IEEE Trans. Evol. Comput., vol. 8, no. 2, pp. 99–110, 2004.
        "MA_S2" : {
            "OM_choice" : "relative_fitness_improv",
            "rew_choice" : "TotalGenAvg",
            "max_gen" : 1,
            "periodic" : False,
            "qual_choice" : "Accumulate",
            "prob_choice" : "ProbabilityMatching",
            "p_min" : 0.0,
            "select_choice" : "ProportionalSelection",
        },
        #  F. Vafaee, P. C. Nelson, C. Zhou, and W. Xiao, “Dynamic adaptation of genetic operators’ probabilities,” in Nature Inspired Cooperative Strategies for Optimization (NICSO 2007), ser. Studies in Computational Intelligence, N. Krasnogor, G. Nicosia, M. Pavone, and D. A. Pelta, Eds. Berlin, Heidelberg: Springer, 2008, vol. 129, pp. 159–168.
        "Dyn_GEPv1" : {
            "OM_choice" : "improv_wrt_parent",
            "rew_choice" : "AvgMetric",
            "periodic" : 0,
            "max_gen" : 1,
            "qual_choice" : "PrevReward",
            "decay_rate" : ([0.01, 1.0], 0.1),
            "q_min" : ([0,1], 0.001),
            "prob_choice" : "ProbabilityMatching",
            "p_min" : 0,
            "select_choice" : "ProportionalSelection",
        },
        "Dyn_GEPv2" : {
            "OM_choice" : "improv_wrt_parent",
            "rew_choice" : "ExtMetric",
            "periodic" : 0,
            "max_gen" : 1,
            "gamma" : 3,
            "qual_choice" : "PrevReward",
            "decay_rate" : ([0.01, 1.0], 0.1),
            "q_min" : ([0,1], 0.001),
            "prob_choice" : "ProbabilityMatching",
            "p_min" : 0,
            "select_choice" : "ProportionalSelection",
        },
        # L. Da Costa, Á. Fialho, M. Schoenauer, and M. Sebag, “Adaptive operator selection with dynamic multi-armed bandits,” in Proceedings of the Genetic and Evolutionary Computation Conference, GECCO 2008, C. Ryan, Ed. New York, NY: ACM Press, 2008, pp. 913–920.
        "DMAB" : {
             "OM_choice" : "improv_wrt_parent",
             "rew_choice" : "AvgMetric",
             "periodic" : 0,
             "max_gen" : 1,
             "qual_choice": "AvgPHrestart",
             "ph_delta": 0.15,
             "ph_threshold": 0,
             "prob_choice": "UCB1",
             "C_ucb" : 0.5,
             "select_choice": "GreedySelection",
         },
        # Á. Fialho, L. Da Costa, M. Schoenauer, and M. Sebag, “Extreme value based adaptive operator selection,” in Parallel Problem Solving from Nature, PPSN X, ser. Lecture Notes in Computer Science, G. Rudolph et al., Eds. Springer, Heidelberg, Germany, 2008, vol. 5199, pp. 175–184.
        "ExMAB" : { 
            "OM_choice" : "improv_wrt_parent",
            "rew_choice" : "ExtAbsWindow",
            "op_winsize" : ([1,100], 50),
            "periodic" : 0,
            "qual_choice" : "AvgPHrestart",
            "ph_delta": 0.15,
            "ph_threshold": 0,
            "prob_choice": "UCB1",
            "C_ucb" : 0.5,
            "select_choice" : "GreedySelection",
        },
        "ExPM" : {
            "OM_choice" : "improv_wrt_parent",
            "rew_choice" : "ExtAbsWindow",
            "op_winsize" : ([1,100], 50),
            "periodic" : 0,
            "qual_choice" : "Qlearning",
            "decay_rate": ([0.01, 1.0], 0.1),
            "q_min" : 0, 
            "prob_choice" : "ProbabilityMatching",
            "p_min" : ([0.0, 1.0], 0.0), # default pmin = 0.0
            "select_choice" : "ProportionalSelection",
        },
        # Á. Fialho, L. Da Costa, M. Schoenauer, and M. Sebag, “Extreme value based adaptive operator selection,” in Parallel Problem Solving from Nature, PPSN X, ser. Lecture Notes in Computer Science, G. Rudolph et al., Eds. Springer, Heidelberg, Germany, 2008, vol. 5199, pp. 175–184.
        "ExtAP" : {
            "OM_choice" : "improv_wrt_parent",
            "rew_choice" : "ExtAbsWindow",
            "op_winsize" : ([1,100], 50),
            "periodic" : 0,
            "qual_choice" : "Qlearning",
            "decay_rate": ([0.01, 1.0], 0.1),
            "q_min" : 0, 
            "prob_choice" : "AdaptivePursuit",
            "p_min" : ([0.0, 1.0], 0.0), # default pmin = 0.0
            "learning_rate": ([0.01, 1.0], 0.1),
            "select_choice" : "ProportionalSelection",
        },
        # Jorge Maturana and Frédéric Saubion. “A compass to guide genetic al-gorithms”. In:International Conference on Parallel Problem Solving from Nature.http://www.info.univ-angers.fr/pub/maturana/files/MaturanaSaubion-Compass-PPSNX.pdf. Springer. 2008, pp. 256–265.
       "Compass" : {
            "OM_choice" : "improv_wrt_parent",
            "rew_choice" : "CompassProjection",
            "app_winsize": 100,
            "theta": ([36, 45, 54, 90], 45),
            "qual_choice" : "QualityIdentity",
            "prob_choice" : "ProbabilityMatching",
            "p_min" : ([0.0, 1.0], 1.0/3.0), # default pmin = 1/3
            "select_choice" : "ProportionalSelection",
        },
    }

    @classmethod
    def build_known_AOS(cls, name,
                        popsize, budget, n_ops, rew_args,
                        qual_args, prob_args, select_args,
                        debug_filename):

        if not name in AOS.known_AOS:
            raise ValueError(f"unknown AOS method {name}, known AOS are {list(AOS.known_AOS)}")
        defaults = AOS.known_AOS[name].copy()
        for ind, value in defaults.items():
            if type(value) == tuple:
                    defaults[ind] = value[1]
        OM_choice = defaults['OM_choice']
        rew_choice = defaults['rew_choice']
        qual_choice = defaults['qual_choice']
        prob_choice = defaults['prob_choice']
        select_choice = defaults['select_choice']
        # Arguments passed overwrite default arguments.
        rew_params = RewardType.params[0::5]
        qual_params = QualityType.params[0::5]
        prob_params = ProbabilityType.params[0::5]
        select_params = SelectionType.params[0::5]
        for ind, value in defaults.items():
            for choice, params in [(rew_args, rew_params),
                                   (qual_args, qual_params),
                                   (prob_args, prob_params),
                                   (select_args, select_params)]:
                if not ind in params:
                    continue
                if not ind in choice or choice[ind] == None:
                    choice[ind] = value
        # For debugging.          
        # print(f'{name} = {dict(OM_choice = OM_choice, rew_choice = rew_choice, rew_args=rew_args, qual_choice=qual_choice, qual_args=qual_args, prob_choice = prob_choice, prob_args=prob_args, select_choice = select_choice, select_args=select_args)}')

        # FIXME: how to use kwargs to avoid having to specify everything that passes through?
        return AOS(popsize, budget, n_ops,
                   OM_choice, rew_choice, rew_args,
                   qual_choice, qual_args,
                   prob_choice, prob_args, select_choice, select_args,
                   debug_filename)

    def __init__(self, popsize, budget, n_ops, OM_choice, rew_choice, rew_args,
                 qual_choice, qual_args, prob_choice, prob_args, select_choice, select_args,
                 debug_filename = None):

        self.n_ops = n_ops
        self.metrics = Metrics(minimize=True, choice=OM_choice)

        self.div_window = None
        self.op_window = None
        self.app_window = None
        # We use the gen_window for statistics, so we always create it.
        self.gen_window = GenWindow(n_ops)
        # Number of applications of each operator (count once per generation, no window)
        self.n_appl = np.zeros(n_ops, dtype=int)

        self.probability = np.full(n_ops, 1.0 / n_ops)
        
        self.periodic = rew_args["periodic"] if rew_args["max_gen"] else False
        self.reward_type = RewardType.build(rew_choice, self, rew_args)
        self.quality_type = QualityType.build(qual_choice, self, qual_args)
        self.probability_type = ProbabilityType.build(prob_choice, self, prob_args)
        self.selection_type = SelectionType.build(select_choice, self, select_args)
        self.update_counter = 1 # We start at 1 so if periodic, we do not update in the first generation.
        
        self.prob_file = DebugFile(debug_filename, ".prob", "p_", n_ops)
        self.rew_file = DebugFile(debug_filename, ".rew", "r_", n_ops)
        self.qual_file = DebugFile(debug_filename, ".qual", "q_", n_ops)

    @classmethod
    def add_arguments(cls, parser):
        Metrics.add_arguments(parser)
        rew_args_names = RewardType.add_arguments(parser)
        qual_args_names = QualityType.add_arguments(parser)
        prob_args_names = ProbabilityType.add_arguments(parser)
        select_args_names = SelectionType.add_arguments(parser)
        return (rew_args_names, qual_args_names, prob_args_names, select_args_names)
        
    @classmethod
    def irace_parameters(cls, override = {}):
        output = "# " + cls.__name__ + "\n"
        output += Metrics.irace_parameters(override = override)
        output += RewardType.irace_parameters(override = override)
        output += QualityType.irace_parameters(override = override)
        output += ProbabilityType.irace_parameters(override = override)
        output += SelectionType.irace_parameters(override = override)
        return output
        

    @classmethod
    def irace_dump_knownAOS(cls, name):
        value = cls.known_AOS[name]
        key = name
        output = "##### AOS:  " + key + ".txt\n"
        output += cls.irace_parameters(override = value)
        return output
        
    @classmethod
    def check_known_AOS(cls):
        has_om = has_rew = has_qual = has_prob = has_sel = False
        for name,parameters in cls.known_AOS.items():
            for param,domain in parameters.items():
                if param == "OM_choice":
                    has_om = True
                    choices = list(Metrics.OM_choices)
                elif param == "rew_choice":
                    has_rew = True
                    choices, _ = get_choices(RewardType)
                elif param == "qual_choice":
                    has_qual = True
                    choices, _ = get_choices(QualityType)
                elif param == "prob_choice":
                    has_prob = True
                    choices, _ = get_choices(ProbabilityType)
                elif param == "select_choice":
                    has_sel = True
                    choices, _ = get_choices(SelectionType)
                else:
                    found = False
                    for x in [RewardType,QualityType,ProbabilityType,SelectionType]:
                        # FIXME: We should also check ranges and conditions. 
                        params = x.params[0::5]
                        found |= param in params
                    assert found,f'unknown {param}={domain} for {name}'
                    continue
                assert np.all(np.isin(domain,choices)),f'unknown {param} {domain} for {name}, {choices}'
            assert has_om,f'{name} does not have OM_metric'
            assert has_rew,f'{name} does not have rew_choice'
            assert has_qual,f'{name} does not have qual_choice'
            assert has_prob,f'{name} does not have prob_choice'
            assert has_sel,f'{name} does not have select_choice'
        
    def select_operator(self):
        return self.selection_type.select(self.probability)

    def update(self, X_p, X_c, F, F1, F_bsf, opu):
        """
        X_p : decision variables of parents
        X_c : decision variables of children
        F: fitness of parent population
        F1: fitness of children population
        F_bsf : best so far fitness
        opu: represents (op)erator that produced offspring (u).
        """
        # To implement Compass-like AOS, we need to store an AppWindow for diversity and another for metric.
        if self.div_window:
            divers_met = calc_diversity(X_p, X_c)
            self.div_window.append(opu, divers_met)
        window_met = self.metrics.update(F1, F, F_bsf)
        if self.op_window:
            self.op_window.append(opu, window_met)
        if self.app_window:
            self.app_window.append(opu, window_met)

        self.gen_window.append(opu, window_met)
        # We count only one application per generation since update is generational
        self.n_appl[np.unique(opu)] += 1
        # Update if we are not doing periodic updates or max_gen == 0 or
        # or counter % max_gen == 0
        if not self.periodic or self.gen_window.max_gen == 0 \
           or (self.update_counter % self.gen_window.max_gen == 0):
            reward = self.reward_type.calc_reward()
            quality = self.quality_type.calc_quality(reward)
            self.probability = self.probability_type.calc_probability(quality)
            # This does nothing unless a debug filename is given.
            self.prob_file.write(self.update_counter, self.probability)
            self.rew_file.write(self.update_counter, reward)
            self.qual_file.write(self.update_counter, quality)
        self.update_counter += 1

class BasicType(ABC):
    # FIXME: Is there a way to call this when defining the class instead of at init?
    @classmethod
    def check_params(cls):
        params = cls.params[0::5]
        choices, _ = get_choices(cls)
        for p,cond in cls.params_conditions.items():
            assert p in params,f'condition key {p} not found in {cls}.params = {params}'
            found = np.isin(cond, choices)
            assert np.all(found),f'condition {cond} not found in {cls} choices = {choices}'
            
    @classmethod
    def add_arguments(cls, parser):
        "Add arguments to an ArgumentParser"
        return parser_add_arguments(cls, parser)

    @classmethod
    def irace_parameters(cls, override = {}):
        return aos_irace_parameters(cls, override = override)

    @classmethod
    def build(cls, choice, aos, kwargs):
        "Build the chosen component given aos and kwargs"
        # Use a dictionary so that we don't have to retype the name.
        choices = get_choices_dict(cls)
        if choice not in choices:
            raise ValueError(f'{cls.__name__} choice {choice} unknown')
        builder = choices[choice]
        for key,cond in cls.params_conditions.items():
            if choice not in cond:
                # We use pop because the key may not exist.
                kwargs.pop(key, None)
        
        return builder(aos = aos, **kwargs)
        

# L. Da Costa, Á. Fialho, M. Schoenauer, and M. Sebag,
# “Adaptive operator selection with dynamic multi-armed
# bandits,” in Proceedings of the Genetic and Evolutionary
# Computation Conference, GECCO 2008, C. Ryan, Ed.
# New York, NY: ACM Press, 2008, pp. 913–920.
class PageHinkleyTest:
    def __init__(self, n_ops, delta, threshold):
        self.delta = delta
        self.threshold = 10**threshold
        self.m = np.zeros(n_ops)
        self.M = np.zeros(n_ops)
    
    def restart(self, reward, quality):
        self.m += (reward - quality + self.delta)
        self.M = np.maximum(self.M, np.abs(self.m))
        do_restart = np.any(self.M - np.abs(self.m) > self.threshold)
        if do_restart:
            self.m[:] = 0
            self.M[:] = 0
        return do_restart
            

class RewardType(BasicType):
    # Static variables
    # FIXME: Use __slots__ to find which parameters need to be defined.
    # FIXME: define this in the class as @property getter doctstring and get it from it
    # MUDITA_check: To generate existiong parameter files, I had to change categorical parameter (theta, succ_lin_quad, normal_factor, alpha, beta, intensity) type to object. Because in irace_parameter() function, categorical is represented as object. But to run this code for parameter tuning, I had to change these categorical to int.
    # FIXME: We do not use these defaults, so that we can override them when using --known-aos. Remove them?
    params = [
        "periodic",       object,      0,     [0, 1], "Periodic update (only every max_gen)",
        "max_gen",          int,        10,     [1, 50], "Maximum number of generations for generational window",
        "app_winsize",      int,        50,     [1, 500],       "Size of application window",
        "op_winsize",      int,        50,     [1, 500],        "Size of window per operator",
        "decay",            float,      0.4,    [0.0, 1.0],     "Decay value to emphasise the choice of better operator",
        "frac",             float,      0.01,   [0.0, 0.1],     "Fraction of sum of successes of all operators",
        "convergence_factor",    float, 20,        [0, 100],    "Factor for convergence credits",
        "gamma",           object,       1,     [1, 2, 3],      "Exponentation (linear, quadratic, ...) of metric values",
        "theta",            object,        45,     [36, 45, 54, 90],               "Compass search direction",
        
    ]
    params_conditions = {
        # Periodic applies to all because it is used by AOS update, but it is
        # not really used explicitly by any method.
        "periodic" : [],
        "max_gen": ["AvgMetric", "NormAvgMetric", "AncestorSuccess", "ExtMetric", "TotalGenAvg", "SuccessRate"],
        "app_winsize": ["AncestorSuccess", "CompassProjection"],
        "decay": ["AncestorSuccess"],
        "frac": ["AncestorSuccess"],
        "convergence_factor": ["AncestorSuccess"],
        "gamma": ["ExtMetric","SuccessRate"],
        "op_winsize": ["AvgAbsWindow", "AvgNormWindow", "ExtAbsWindow", "ExtNormWindow"],
        "theta" : ["CompassProjection"],
    }
    param_choice = "rew_choice"
    param_choice_help = "Reward method selected"

    def __init__(self, aos, max_gen = None, app_winsize = None, op_winsize = None, decay = None):
        # Set a few common short-cuts.
        self.aos = aos
        self.n_ops = aos.n_ops
        self.metrics = aos.metrics
        if max_gen:
            self.gen_window = aos.gen_window
            self.gen_window.max_gen = max_gen
        if app_winsize:
            self.aos.app_window = AppWindow(self.n_ops, max_size = app_winsize)
        if op_winsize:
            self.aos.op_window = OpWindow(self.n_ops, max_size = op_winsize)
        self.decay = decay

    def check_reward(self, reward):
        # rew_min = reward.min()
        # rew_diff = reward.max() - rew_min
        # if rew_diff > 0:
        #     reward = (reward - rew_min) / rew_diff
        # else:
        #     reward[:] = 0.0
        assert np.all(np.isfinite(reward)), f"Infinite reward {reward}"
        assert np.all(reward >= 0), f"Negative reward {reward}"
        return reward

    @abstractmethod
    def calc_reward(self):
        pass


class AvgMetric(RewardType):

    def __init__(self, aos, max_gen = 4):
        super().__init__(aos, max_gen = max_gen)
        assert max_gen > 0
        debug_print(f"{type(self).__name__}: max_gen = {self.gen_window.max_gen}")

    def calc_reward(self):
        _, napplications = self.gen_window.count_succ_total()
        napplications[napplications == 0] = 1
        reward = self.gen_window.sum_per_op()
        reward /= napplications
        return super().check_reward(reward)

class NormAvgMetric(AvgMetric):
    """C. Igel and M. Kreutz, “Operator adaptation in evolutionary computation and its application to structure optimization of neural networks,” Neurocomputing, vol. 55, no. 1-2, pp. 347–361, 2003.
    """

    def __init__(self, aos, max_gen = 4):
        super().__init__(aos, max_gen = max_gen)
        assert max_gen > 0
        debug_print(f"{type(self).__name__}: max_gen = {self.gen_window.max_gen}")

    def calc_reward(self):
        reward = super().calc_reward()
        # The paper also normalizes by the total but this could be optional or
        # it could be done in the quality component.
        reward = normalize_sum(reward)        
        return super().check_reward(reward)


class ExtMetric(RewardType):
    """F. G. Lobo and D. E. Goldberg, “Decision making in a hybrid genetic
algorithm,” in Proceedings of the 1997 IEEE International Conference on
Evolutionary Computation (ICEC’97), T. Bäck, Z. Michalewicz, and X. Yao,
Eds. Piscataway, NJ: IEEE Press, 1997, pp.  121–125."""
    
    def __init__(self, aos, max_gen = 10000, gamma = 1):
        super().__init__(aos, max_gen = max_gen)
        self.gamma = int(gamma)
        assert self.gamma >= 1 and self.gamma <= 4
        # When the window is large, we speed up computation by tracking the max
        # ourselves.  This breaks the sliding window, but for such large
        # windows it doesn't really make sense to worry about that.
        self.use_window = max_gen < 10000
        self.max_metric = np.zeros(self.n_ops)
        debug_print(f"{type(self).__name__}: max_gen = {self.gen_window.max_gen}  gamma = {self.gamma}")
    
    def calc_reward(self):
        max_gen = self.gen_window.get_max_gen()
        last_gen = len(self.gen_window) - 1
        if self.use_window or max_gen < last_gen:
            gen_window_len = len(self.gen_window)
            # We override the saved value if we use the sliding window.
            self.max_metric = self.gen_window.max_per_op()
        else:
            # If the window is anyway not full, use pre-computed.
            self.max_metric = np.maximum(self.max_metric,
                                         self.gen_window.max_per_op(last_gen))
        reward = self.max_metric ** self.gamma
        return super().check_reward(reward)


class SuccessRate(RewardType):
    """ 
    J. Niehaus and W. Banzhaf, “Adaption of operator
    probabilities in genetic programming,” in Proceedings of
    the 4th European Conference on Genetic Programming,
    EuroGP 2001, ser. LNCS, J. Miller, M. Tomassini, P. L.
    Lanzi, C. Ryan, A. G. B. Tettamanzi, and W. B. Langdon,
    Eds. Springer, 2001, vol. 2038, pp. 325–336.
"""
    
    def __init__(self, aos, max_gen = 10000, gamma = 2):
        super().__init__(aos, max_gen = max_gen)
        self.gamma = int(gamma)
        assert self.gamma >= 1 and self.gamma <= 4
        # When the window is large or not full, we speed up computation by
        # keeping our own counts.  This breaks the sliding window, but for such
        # large windows it doesn't really make sense to worry about that.
        self.use_window = max_gen < 10000
        self.success = np.zeros(self.n_ops)
        self.used = np.ones(self.n_ops)
        debug_print(f"{type(self).__name__}: max_gen = {self.gen_window.max_gen}  gamma = {self.gamma}")
            
    def calc_reward(self):
        max_gen = self.gen_window.get_max_gen()
        last_gen = len(self.gen_window) - 1
        if self.use_window or max_gen < last_gen:
            self.success, self.used = self.gen_window.count_succ_total()
        else:
            # If the window is anyway not full, use pre-computed.
            success, used = self.gen_window.count_succ_total(last_gen)
            self.success += success
            self.used += used
        reward = (self.success ** self.gamma) / self.used
        return super().check_reward(reward)


class AncestorSuccess(RewardType):
    '''
    B. A. Julstrom, "What have you done for me lately? adapting operator probabilities in a steady-state genetic algorithm," in ICGA, L. J. Eshelman, Ed. Morgan Kaufmann Publishers, San Francisco, CA, 1995, pp. 81–87.

    B. A. Julstrom, "An inquiry into the behavior of adaptive operator probabilities in steady-state genetic algorithms,” in Proceedings of the Second Nordic Workshop on Genetic Algorithms and their Applications, August, 1996, pp. 15–26.
    '''
    def __init__(self, aos, app_winsize = 100, max_gen = 5, decay = 0.8, frac = 0.01, convergence_factor = 20):
        super().__init__(aos, app_winsize = app_winsize, max_gen = max_gen)
        assert decay >= 0.0 and decay <= 1.0
        # The most recent generation is max_gen - 1.
        self.decay_g = decay ** np.arange(max_gen - 1, -1, -1)
        assert frac >= 0.0 and frac <= 1.0
        self.frac = frac

        assert convergence_factor >= 0
        self.convergence_factor = convergence_factor
        debug_print(f"{type(self).__name__}: app_winsize = {app_winsize}  max_gen = {self.gen_window.max_gen}  decay_g = {self.decay_g}  frac = {self.frac}  convergence_factor = {self.convergence_factor}")
        # In the paper, we use the application window, however, it is more
        # efficient to just memorise the values as a queue
        self.accum_credit = deque(maxlen = app_winsize)
        self.total_credit = deque(maxlen = app_winsize)
        
    def calc_reward(self):
        reward = np.zeros(self.n_ops)

        last_gen = len(self.gen_window) - 1
        succ = self.gen_window.is_success(last_gen)
        which_succ = np.where(succ)[0]
        for i in which_succ:
            # For each child i, we get the operators applied to it
            ops = self.gen_window.get_ops_of_child(i)
            # We get the decay that corresponds to each operator.
            D = self.decay_g[-len(ops):]
            # For each operator, its reward is the sum of decays.
            for op in range(self.n_ops):
                reward[op] += D.sum(where = (op ==  ops))

        self.accum_credit.append(reward)
        tot_acc_reward = np.sum(self.accum_credit, axis=0)
        if self.frac > 0:
            self.total_credit.append(reward.sum())
            tot_acc_reward += self.frac * sum(self.total_credit)

        if self.convergence_factor > 0:
            diff = self.metrics.F_best - self.metrics.F_median
            tmp = (1.0 / (abs(diff)**3)) if diff > 0 else 2
            tot_acc_reward += self.convergence_factor * tmp
            
        ops_count = self.aos.app_window.count_ops()
        tot_acc_reward[ops_count == 0] = 0
        ops_count[ops_count == 0] = 1
        reward = tot_acc_reward / ops_count
        return super().check_reward(reward)

class TotalGenAvg(RewardType): # This was Normalised_success_sum_gen
    """C. Igel and M. Kreutz, “Using fitness distributions to improve the evolution of learning structures,” in Proceedings of the 1999 Congress on Evolutionary Computation (CEC 1999), vol. 3. Piscataway, NJ: IEEE Press, 1999, pp. 1902–1909.
"""
    def __init__(self, aos, max_gen = 4):
        super().__init__(aos, max_gen = max_gen)
        assert max_gen > 0
        debug_print(f"{type(self).__name__}: max_gen = {self.gen_window.max_gen}")
    
    def calc_reward(self):
        # Most recent generation
        gen_window_len = len(self.gen_window)
        # Oldest generation
        max_gen = self.gen_window.get_max_gen()
        reward = np.zeros(self.n_ops)
        # FIXME: We could have a gen_window.mean_per_generation()
        for g in range(gen_window_len - max_gen, gen_window_len):
            value = self.gen_window.sum_per_op(g)
            _, napplications = self.gen_window.count_succ_total(g)
            reward += value / napplications
        return super().check_reward(reward)


class AvgAbsWindow(RewardType):
    """
    Á. Fialho, M. Schoenauer, and M. Sebag. Analysis of adaptive operator selection techniques on the royal road and long k-path problems. In Proc. Genetic Evol. Comput. Conf., pages 779–786, 2009.
    """
    def __init__(self, aos, op_winsize = 50):
        super().__init__(aos, op_winsize = op_winsize)
        debug_print(f"{type(self).__name__}: op_winsize = {self.op_winsize}")

    def calc_reward(self):
        reward = self.aos.op_window.mean_per_op()
        return super().check_reward(reward)

class AvgNormWindow(AvgAbsWindow):
    """
    Á. Fialho, M. Schoenauer, and M. Sebag. Analysis of adaptive operator selection techniques on the royal road and long k-path problems. In Proc. Genetic Evol. Comput. Conf., pages 779–786, 2009.
    """
    def __init__(self, aos, op_winsize = 50):
        super().__init__(aos, op_winsize = op_winsize)
        debug_print(f"{type(self).__name__}: op_winsize = {self.op_winsize}")

    def calc_reward(self):
        reward = super().calc_reward()
        reward = normalize_max(reward)
        return super().check_reward(reward)

class ExtAbsWindow(RewardType):
    """
    Á. Fialho, M. Schoenauer, and M. Sebag. Analysis of adaptive operator selection techniques on the royal road and long k-path problems. In Proc. Genetic Evol. Comput. Conf., pages 779–786, 2009.
    """
    def __init__(self, aos, op_winsize = 50):
        super().__init__(aos, op_winsize = op_winsize)
        debug_print(f"{type(self).__name__}: op_winsize = {self.op_winsize}")

    def calc_reward(self):
        reward = self.aos.op_window.max_per_op()
        return super().check_reward(reward)

class ExtNormWindow(ExtAbsWindow):
    """
    Á. Fialho, M. Schoenauer, and M. Sebag. Analysis of adaptive operator selection techniques on the royal road and long k-path problems. In Proc. Genetic Evol. Comput. Conf., pages 779–786, 2009.
    """
    def __init__(self, aos, op_winsize = 50):
        super().__init__(aos, op_winsize = op_winsize)
        debug_print(f"{type(self).__name__}: op_winsize = {self.op_winsize}")

    def calc_reward(self):
        reward = super().calc_reward()
        reward = normalize_max(reward)
        return super().check_reward(reward)

class CompassProjection(RewardType):
    """
        Jorge Maturana and Frédéric Saubion. “A compass to guide genetic algorithms”. In:International Conference on Parallel Problem Solving fromNature.http://www.info.univ-angers.fr/pub/maturana/files/MaturanaSaubion-Compass-PPSNX.pdf. Springer. 2008, pp. 256–265.
        """
    def __init__(self, aos, app_winsize = 100, theta = 45):
        super().__init__(aos, app_winsize = 100)
        self.aos.div_window = AppWindow(self.n_ops, app_winsize)
        debug_print(f"{type(self).__name__}: app_winsize = {self.app_winsize}  theta = {theta}")
        self.theta = np.deg2rad(theta)
    
    def calc_reward(self):
        avg_div = self.aos.div_window.mean_per_op()
        avg_div = normalise_max(avg_div)
        avg_met = self.aos.app_window.mean_per_op()
        avg_met = normalise_max(avg_met)
        # ||(d,m)||
        norm = np.sqrt(avg_dist**2 + avg_met**2)
        angle = np.abs(np.arctan2(avg_met, avg_div) - self.theta)
        reward = norm * np.cos(angle)
        reward = reward - reward.min()
        # Maturana & Sablon (2008) divide by T_it defined as mean execution
        # time of operator i over its last t applications. We do not handle execution time yet.
        return super().check_reward(reward)

class QualityType(BasicType):
    # Static variables
    params = [
        "decay_rate",        float, 0.6,    [0.01, 1.0],     "Decay rate (delta)",
        "q_min",             float, 0.1,    [0.0, 1.0],     "Minimum quality attained by an operator (divided by num operators)",
        "ph_delta",          float, 0.15,   [0.01, 1.0],    "Delta for Page-Hinkley test",
        "ph_threshold",      int,   0,      [-3, 3],    "10^threshold for Page-Hinkley test"
    ]
    params_conditions = {
        "decay_rate" : ["Qlearning", "Qdecay", "PrevReward"],
        "q_min" : ["Qlearning", "Qdecay", "PrevReward"],
        "ph_delta" : ["AvgPHrestart"],
        "ph_threshold" : ["AvgPHrestart"]
 
    }
    param_choice = "qual_choice"
    param_choice_help = "Quality method selected"
    
    def __init__(self, aos, decay_rate = None, q_min = None):
        self.aos = aos
        self.n_ops = aos.n_ops
        self.old_quality = np.zeros(self.n_ops)
        if decay_rate != None:
            assert decay_rate >= 0 and decay_rate <= 1
            self.decay_rate = decay_rate
        if q_min != None:
            assert q_min >= 0 and q_min <= 1
            self.q_min = q_min / self.n_ops
 
    def check_quality(self, quality):
        assert np.sum(quality) >= 0
        # FIXME: why do we need to normalize it to sum to 1?
        quality = normalize_sum(quality)
        self.old_quality[:] = quality
        return quality
    
    @abstractmethod
    def calc_quality(self, reward):
        pass

class QualityIdentity(QualityType):
    def __init__(self, aos):
        super().__init__(aos)
        debug_print(f"{type(self).__name__}: n_ops = {self.n_ops}")
        
    def calc_quality(self, reward):
        return self.check_quality(reward)

### Previously Weighted_sum
class Qlearning(QualityType):
    """ """
    def __init__(self, aos, decay_rate = 0.6, q_min = 0.0):
        super().__init__(aos, decay_rate = decay_rate, q_min = q_min)
        debug_print(f"{type(self).__name__}: n_ops = {self.n_ops}  decay_rate = {self.decay_rate}  q_min = {self.q_min}")

    def calc_quality(self, reward):
        quality = self.decay_rate * np.maximum(self.q_min, reward) + (1.0 - self.decay_rate) * self.old_quality
        return super().check_quality(quality)

class Qdecay(QualityType):
    """ """
    def __init__(self, aos, decay_rate = 0.6, q_min = 0.0):
        super().__init__(aos, decay_rate = decay_rate, q_min = q_min)
        debug_print(f"{type(self).__name__}: n_ops = {self.n_ops}  decay_rate = {self.decay_rate}  q_min = {self.q_min}")
    
    def calc_quality(self, reward):
        quality = self.q_min + reward + (1.0 - self.decay_rate) * self.old_quality
        return super().check_quality(quality)

class Accumulate(QualityType):
    def __init__(self, aos):
        super().__init__(aos)
        debug_print(f"{type(self).__name__}: n_ops = {self.n_ops}")
        
    def calc_quality(self, reward):
        quality = reward + self.old_quality
        return self.check_quality(quality)

class PrevReward(QualityType):
    """ """
    def __init__(self, aos, decay_rate = 0.6, q_min = 0.0):
        super().__init__(aos, decay_rate = decay_rate, q_min = q_min)
        self.old_reward = np.zeros(self.n_ops)
        debug_print(f"{type(self).__name__}: n_ops = {self.n_ops}  decay_rate = {self.decay_rate}  q_min = {self.q_min}")
    
    def calc_quality(self, reward):
        # It basically ignores self.old_quality
        quality = self.q_min + reward + (1.0 - self.decay_rate) * self.old_reward
        self.old_reward[:] = reward
        return self.check_quality(quality)

# L. Da Costa, Á. Fialho, M. Schoenauer, and M. Sebag,
# “Adaptive operator selection with dynamic multi-armed
# bandits,” in Proceedings of the Genetic and Evolutionary
# Computation Conference, GECCO 2008, C. Ryan, Ed.
# New York, NY: ACM Press, 2008, pp. 913–920.
class PageHinkleyTest:
    def __init__(self, n_ops, delta, threshold):
        self.delta = delta
        self.threshold = 10**threshold
        self.m = np.zeros(n_ops)
        self.M = np.zeros(n_ops)
    
    def restart(self, reward, quality):
        self.m += (reward - quality + self.delta)
        self.M = np.maximum(self.M, np.abs(self.m))
        do_restart = np.any(self.M - np.abs(self.m) > self.threshold)
        if do_restart:
            self.m[:] = 0
            self.M[:] = 0
        return do_restart

class AvgPHrestart(QualityType):
    """L. Da Costa, Á. Fialho, M. Schoenauer, and M. Sebag, “Adaptive operator selection with dynamic multi-armed bandits,” in Proceedings of the Genetic and Evolutionary Computation Conference, GECCO 2008, C. Ryan, Ed.
 New York, NY: ACM Press, 2008, pp. 913–920.
"""
    def __init__(self, aos, ph_delta, ph_threshold):
        super().__init__(aos)
        assert ph_delta > 0
        self.ph_test = PageHinkleyTest(self.n_ops, delta = ph_delta, threshold = ph_threshold)
        debug_print(f"{type(self).__name__}: n_ops = {self.n_ops}  ph_delta = {self.ph_test.delta}  ph_threshold = {self.ph_test.threshold}")
    
    def calc_quality(self, reward):
        inv_ntot = 1. / np.where(self.aos.n_appl > 0, self.aos.n_appl, 1)
        quality = inv_ntot * reward + (1. - inv_ntot) * self.old_quality
        if self.ph_test.restart(reward, quality):
            debug_print(f"{type(self).__name__}: PH restart")
            self.aos.n_appl[:] = 0
            quality[:] = 0
        return super().check_quality(quality)

    
class ProbabilityType(BasicType):
    # Static variables
    params = [
        "p_min",         float,     0,    [0.0, 1], "Minimum probability of selection of an operator is p_min / n_ops)",
        "learning_rate", float,     0.1,    [0.0, 1.0], "Learning Rate",
        "C_ucb",         float,     0.5,    [0.01, 100.0], "Scaling factor in UCB",

    ]
    params_conditions = {
        "p_min": ["ProbabilityMatching", "AdaptivePursuit"],
        "learning_rate" : ["AdaptivePursuit"],
        "C_ucb": ["UCB1"],
    }
    param_choice = "prob_choice"
    param_choice_help = "Probability method selected"
        
    def __init__(self, aos, p_min = 0, learning_rate = None):
        n_ops = aos.n_ops
        assert p_min >= 0 and p_min <= 1
        self.p_min = p_min / n_ops
        self.one_minus_p_min = (1.0 - p_min)
        assert self.one_minus_p_min > 0.0
        self.learning_rate = learning_rate
        self.old_probability = np.full(n_ops, 1.0 / n_ops)
        
    def check_probability(self, probability):
        assert np.all(probability >= 0.0)
        probability = normalize_sum(probability)
        assert np.allclose(probability.sum(), 1.0, equal_nan = True),f'prob = {probability}'
        # Just copy the values.
        self.old_probability[:] = probability
        return probability

    @abstractmethod
    def calc_probability(self, quality):
        "Must be implemented by derived probability methods"
        pass

class ProbabilityMatching(ProbabilityType):
    # FIXME: Probability matching predates this paper.
    """Dirk Thierens. "An adaptive pursuit strategy for allocating operator probabilities".  In: Proceedings of the 7th annual conference on Genetic and evolutionary computation. http://www.cs.bham.ac.uk/~wbl/biblio/gecco2005/docs/p1539.pdf. ACM. 2005, pp. 1539–1546."""
    
    def __init__(self, aos, p_min = 0.1):
        super().__init__(aos, p_min = p_min)
        debug_print(f"{type(self).__name__}: n_ops = {aos.n_ops}  p_min = {self.p_min}")
                
    def calc_probability(self, quality):
        probability = normalize_sum(quality)
        # self.p_min is already divided by n_ops
        if self.p_min > 0: 
            probability = self.p_min + self.one_minus_p_min * probability
        return self.check_probability(probability)

class AdaptivePursuit(ProbabilityType):
    """ Proposed by:
Dirk Thierens. “An adaptive pursuit strategy for allocating operator prob-
abilities”. In: Proceedings of the 7th annual conference on Genetic and
evolutionary computation. http://www.cs.bham.ac.uk/~wbl/biblio/gecco2005/docs/p1539.pdf. ACM. 2005, pp. 1539–1546.

"""
    def __init__(self, aos, p_min = 0.1, learning_rate = 0.1):
        super().__init__(aos, p_min = p_min, learning_rate = learning_rate)
        self.p_max = (aos.n_ops - 1) * self.p_min
        assert self.p_max > self.p_min
        debug_print(f"{type(self).__name__}: n_ops = {aos.n_ops}  p_min = {self.p_min}  p_max = {self.p_max}  learning_rate = {self.learning_rate}")
        
    def calc_probability(self, quality):
        delta = np.full(len(quality), self.p_min)
        delta[np.argmax(quality)] = self.p_max
        probability = self.learning_rate * delta + (1.0  - self.learning_rate) * self.old_probability
        return super().check_probability(probability)

class LinearRank(ProbabilityType):
    """Probability proportional to rank, not value."""
    def __init__(self, aos):
        super().__init__(aos)
        debug_print(f"{type(self).__name__}: n_ops = {aos.n_ops}")
        
    def calc_probability(self, quality):
        probability = rankdata(quality, method = "min")
        # check_probability normalizes to sum already.
        return super().check_probability(probability)

class UCB1(ProbabilityType):
    '''L. Da Costa, Á. Fialho, M. Schoenauer, and M. Sebag, “Adaptive operator selection with dynamic multi-armed bandits,” in Proceedings of the Genetic and Evolutionary Computation Conference, GECCO 2008, C. Ryan, Ed. New York, NY: ACM Press, 2008, pp. 913–920.'''
    def __init__(self, aos, C_ucb = 2):
        super().__init__(aos)
        n_ops = aos.n_ops
        assert C_ucb > 0
        self.C_ucb = C_ucb
        self.used = np.ones(n_ops)
        self.ph_m = np.zeros(n_ops)
        self.ph_M = np.zeros(n_ops)
        
    def calc_probability(self, quality):
        '''Calculates Upper Confidence Bound (1)'''
        n = self.aos.np_appl
        n[n == 0] = 1
        probability = quality + self.C_ucb * np.sqrt(2 * np.log(n.sum()) / n)
        return super().check_probability(probability)


class SelectionType(BasicType):
    params = [
    ]
    params_conditions = {
    }

    param_choice = "select_choice"
    param_choice_help = "Selection method"
        
    def __init__(self, aos):
        # The initial list of operators (randomly permuted)
        self.n_ops = aos.n_ops
        self.op_init_list = list(np.random.permutation(self.n_ops))

    def check_selection(self, selected):
        assert selected >= 0 and selected <= self.n_ops
        return selected

    # Python 3.8 adds @final
    def select(self, probability):
        if self.op_init_list:
            op = self.op_init_list.pop()
        else:
            op = self._select(probability)
        return self.check_selection(op)
        
    @abstractmethod
    def _select(self, probability, update_counter):
        pass

class ProportionalSelection(SelectionType):
    """Also called Roulette wheel selection.

 Thierens, Dirk. "An adaptive pursuit strategy for allocating operator probabilities." Proceedings of the 7th annual conference on Genetic and evolutionary computation. ACM, 2005. """
    def __init__(self, aos):
        super().__init__(aos)
    
    def _select(self, probability):
        # Roulette wheel selection
        return np.random.choice(len(probability), p = probability)

class GreedySelection(SelectionType):
    """Fialho, Álvaro, Marc Schoenauer, and Michèle Sebag. "Toward comparison-based adaptive operator selection." Proceedings of the 12th annual conference on Genetic and evolutionary computation. ACM, 2010.
"""
    def __init__(self, aos):
        super().__init__(aos)
    
    def _select(self, probability):
        # Greedy Selection
        return np.argmax(probability)

# FIXME: Find a way to call these checks at class creation time.
# Sanity checks.
RewardType.check_params()
QualityType.check_params()
ProbabilityType.check_params()
SelectionType.check_params()
AOS.check_known_AOS()
