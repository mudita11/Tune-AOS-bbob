# FIXME: Find all non-ascii characters and replace them for their ASCII equivalent.
from __future__ import print_function
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

def debug_print(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def normalize_sum(x):
    # The divide by zero check does not work if we have negative values because
    # [5, -5] sums to zero.
    assert np.all(x >= 0.0)
    s = x.sum()
    if s != 0: return x / s
    return x

def softmax(x):
    """Normalises each real value in a vector in the range 0 and 1 (normalised exponential function)"""
    return np.exp(x - x.max())

def get_choices(cls, override = []):
    """Get all possible choices of a component of AOS framework"""
    choices = [x.__name__ for x in cls.__subclasses__()]
    if len(override):
        choices = override
    choices_help = ', '.join(f"{i}" for i in choices)
    return choices, choices_help
    
    
def parser_add_arguments(cls, parser):
    "Helper function to add arguments of a class to an ArgumentParser"
    choices, choices_help = get_choices(cls)
    group = parser.add_argument_group(title=cls.__name__)
    group.add_argument("--"  + cls.param_choice, choices=choices,
                       help=cls.param_choice_help + " (" + choices_help + ")")

    for i in range(0, len(cls.params), 5):
        arg, type, default, domain, help = cls.params[i:i+5]
        if type == object:
            type = str
        #group.add_argument('--' + arg, type=type, default=default, help=help)
        group.add_argument('--' + arg, type=type, default=None, help=help)
    # Return the names
    return cls.params[0::5]

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

    arg = f'"--{name} "'
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
        
    if len(values) == 1:
        return what + " == " + as_r_string(values[0])
    return what + " %in% c(" + ", ".join([as_r_string(x) for x in values]) + ")"


class GenWindow(object):
    ''''Generational Window of OM values. g=0 is the oldest, while g=len(self)-1 is the most recent generation'''
# FIXME (needs updating): gen_window stores the offspring metric data for each offspring when offspring is better than parent. Otherwise it stores np.nan for that offspring. Its a list. Its structre is as follows: [[[second_dim], [second_dim], [second_dim]], [[],[],[]], ...]. Second_dim represnts Offspring metric data for an offspring. The number of second dims will be equal to the population size, contained in third_dim. Third_dim represents a generation. Thus, [[],[],[]] has data of all offsprings in a generation."""
    
    def __init__(self, n_ops, metric, max_gen = 0):
        # Private
        self.n_ops = n_ops
        self.metric = metric
        self.max_gen = max_gen
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
        
        
    def apply_at_generation(self, gen, function):
        """Apply function to metric values at generation gen for all operators"""
        window_met = self._gen_window_met[gen, :, self.metric]
        window_op = self._gen_window_op[gen, :]
        value = np.zeros(self.n_ops)
        is_not_nan = ~np.isnan(window_met)
        for op in range(self.n_ops):
            temp_window_met = np.zeros(len(window_op))
            # Assign 0.0 to any entry that is nan or belongs to a different op
            temp_window_met = np.where((window_op == op) & is_not_nan, window_met, 0.0)
            value[op] = function(temp_window_met)
        return value

    def sum_at_generation(self,gen):
        """Get metric sum for all operators at generation gen"""
        return self.apply_at_generation(gen, np.sum)
    
    def max_at_generation(self, gen):
        """Get best metric value for all operators at generation gen"""
        return self.apply_at_generation(gen, np.max)
        
    def max_per_generation(self, op):
        """Get best metric value for operator op for each of the last max_gen generations"""
        gen_window_len = len(self)
        max_gen = self.get_max_gen()
        start = gen_window_len - max_gen
        window_met = self._gen_window_met[start:, :, self.metric]
        window_op = self._gen_window_op[start:, :]
        # Assign 0.0 to any entry that is nan or belongs to a different op
        window_met = np.where((window_op == op) & ~np.isnan(window_met), window_met, 0.0)
        assert window_met.shape[0] == max_gen
        # maximum per row, as many rows as max_gen
        return np.max(window_met, axis = 1)

    def is_success(self, gen):
        window_met = self._gen_window_met[gen, :, self.metric]
        return ~np.isnan(window_met)
        
    def total_success(self):
        window_met = self._gen_window_met[-self.max_gen:, :, self.metric].ravel()
        window_op = self._gen_window_op[-self.max_gen:, :].ravel()
        is_succ = ~np.isnan(window_met)
        total_success = np.zeros(self.n_ops)
        total_apps = np.zeros(self.n_ops)
        for op in range(self.n_ops):
            is_op = window_op == op
            total_success[op] = np.sum(is_op & is_succ)
            total_apps[op] = np.sum(is_op)
        total_apps[total_apps == 0] = 1.
        return total_success, total_apps

    def success(self, gen):
        window_met = self._gen_window_met[gen, :, self.metric]
        window_op = self._gen_window_op[gen, :]
        is_succ = ~np.isnan(window_met)
        total_success = np.zeros(self.n_ops)
        total_apps = np.zeros(self.n_ops)
        for op in range(self.n_ops):
            is_op = window_op == op
            total_success[op] = np.sum(is_op & is_succ)
            total_apps[op] = np.sum(is_op)
        total_apps[total_apps == 0] = 1.
        total_success /= total_apps
        return total_success, total_apps

    def count_total_succ_unsucc(self, gen):
        """Counts the number of successful and unsuccessful applications for each operator in generation 'gen'"""
        window_met = self._gen_window_met[gen, :, self.metric]
        window_op = self._gen_window_op[gen, :]
        total_success = np.zeros(self.n_ops)
        total_unsuccess = np.zeros(self.n_ops)
        is_nan = np.isnan(window_met)
        is_not_nan = ~is_nan
        for op in range(self.n_ops):
            is_op = window_op == op
            total_success[op] = np.sum(is_op & is_not_nan)
            total_unsuccess[op] = np.sum(is_op & is_nan)
        return total_success, total_unsuccess

    def metric_for_fix_appl_of_op(self, op, fix_appl):
        """Return a vector of metric values for last fix_appl applications of operator op"""
        # Stop at fix_appl starting from the end of the window (latest fix_applications of operators)
        # MUDITA_check: Whats the use of gen_window_len here?
        gen_window_len = len(self)
        window_met = self._gen_window_met[:, :, self.metric]
        window_op = self._gen_window_op[:, :]
        # Without np.where, it returns a 1D array
        b = window_met[(window_op == op) & ~np.isnan(window_met)]
        # Keep only the last fix_appl values
        return b[-fix_appl:]
        
    def write_to(self, filename):
        ops = self._gen_window_op
        met = self._gen_window_met
        gen = np.tile(np.arange(ops.shape[0]).reshape(-1,1), (1, ops.shape[1]))
        out = np.hstack((ops.reshape(-1,1),
                         gen.reshape(-1,1),
                         met.reshape(met.shape[0] * met.shape[1], met.shape[2])))
        np.savetxt(filename, out, fmt= 2*["%d"] + 7*["%+20.15e"],
                   header = "operator generation"
                   + " " + "absolute_fitness"
                   + " " + "exp_absolute_fitness"
                   + " " + "improv_wrt_parent"
                   + " " + "improv_wrt_pop"
                   + " " + "improv_wrt_bsf"
                   + " " + "improv_wrt_median"
                   + " " + "relative_fitness_improv")
                   
class OpWindow(object):

    def __init__(self, n_ops, metric, max_size = 0):
        self.max_size = max_size
        self.n_ops = n_ops
        self.metric = metric
        # Vector of operators
        self._window_op = np.full(max_size, -1)
        # Matrix of metrics
        # np.inf means not initialized
        # np.nan means unsuccessful application
        self._window_met = np.full((max_size, len(AOS.OM_choices)), np.inf)
                
    def resize(self, max_size):
        self.max_size = max_size
        # Vector of operators
        self._window_op = np.full(max_size, -1)
        self._window_met = np.full((max_size, len(AOS.OM_choices)), np.inf)
    
    def count_ops(self):
        N = np.zeros(self.n_ops)
        op, count = np.unique(self._window_op, return_counts=True)
        N[op] = count
        return N
    
    def truncate(self, size):
        where = self.where_truncate(size)
        # MUDITA: truncated working is not clear.
        truncated = copy.copy(self)
        truncated._window_op = truncated._window_op[where]
        truncated._window_met = truncated._window_met[where, :]
        return truncated
    
    def where_truncate(self, size):
        """Returns the indexes of a truncated window after removing the offspring entry with unimproved metric from window and truncating to size"""
        assert size > 0
        # np.where returns a tuple, use np.flatnonzero to return a 1D array
        where = np.flatnonzero(np.isfinite(self._window_met[:, self.metric]))
        return where[:size]

    def sum_per_op(self):
        # FIXME: there is probably a faster way to do this.
        #met_values = self._window_met[:, self.metric]
        #return np.bincount(self._window_op, weights = met_values, minlength = self.n_ops)
        value = np.zeros(self.n_ops)
        for i in range(self.n_ops):
            value[i] = np.sum(self._window_met[self._window_op == i, self.metric])
        return value
    
    def get_ops_sorted_and_rank(self):
        '''Return sorted window, number of successful applications of operators and rank'''
        assert np.all(np.isfinite(self._window_met[:, self.metric]))
        assert np.all(self._window_op >= 0)
        x = self._window_met[:, self.metric]
        # Gives rank to window[:, off_met]: largest number will get smallest number rank.
        rank = rankdata(-x, method="min")
        order = rank.argsort()
        # If rank is [3, 1, 2]; then order of rank will be [1, 2, 0] because value ranked 3 is present at index 0. Thus, order[3] = 0 or order[rank] = index of rank.
        # window_op_sorted is the operator vector sorted according to order i.e. highest Off_metrix to lowest
        window_op_sorted = self._window_op[order]
        rank = rank[order]
        assert len(window_op_sorted) == len(rank)
        return window_op_sorted, rank

    def append(self, op, values):
        '''Push data of improved offspring in the window. It follows First In First Out Rule.'''
        # Fill from the bottom
        which = (np.isinf(self._window_met[:,1]))
        if np.any(which):
            last_empty = np.max(np.flatnonzero(which))
            self._window_op[last_empty] = op
            self._window_met[last_empty, :] = values
            return

        # Find last element that matches op
        which = (self._window_op == op)
        if np.any(which):
            last = np.max(np.flatnonzero(which))
        else:
            # If the operator is not in the window, remove the worst if it is
            # worse than the value we want to add.
            last = np.argmin(self._window_met[:, 0])
            if self._window_met[last, 0] >= values[0]:
                return

        # Shift contents of window
        self._window_op[1:(last+1)] = self._window_op[0:last]
        self._window_met[1:(last+1), :] = self._window_met[0:last, :]
        # Add it to the top
        self._window_op[0] = op
        self._window_met[0, :] = values

class Metrics(object):

    eps = np.finfo(np.float32).eps

    def __init__(self, minimize):
        # FIXME: Extend to handle both minimization and maximization.
        assert minimize == True
        self.upper_bound = None
        
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
                
    # Fitness is minimised but metric is maximised
    def absolute_fitness(self):
        if self.upper_bound is None:
            # If all values are negative, then we need abs()
            self.upper_bound = abs(self.F_children.max()) * 10
        # If a later individual is still worse than the upper bound, it is
        # probably very bad anyway.
        values = np.maximum(0, self.upper_bound - self.F_children)
        assert (values >= 0).all(),f'F_children = {self.F_children[values <= 0]}, U = {self.upper_bound}'
        return values

    def exp_absolute_fitness(self):
        return softmax(-self.F_children)
    # FIXME: In the paper we say max(0, improvement), why not here?
    def improv_wrt_parent(self):
        return self.F_parents - self.F_children
    def improv_wrt_pop(self):
        return self.F_best - self.F_children
    def improv_wrt_bsf(self):
        return self.F_bsf - self.F_children
    def improv_wrt_median(self):
        return self.F_median - self.F_children
    def relative_fitness_improv(self):
        # MUDITA: if F_bsf = 1 and F1 = 0, then F_bsf / (F1 + eps) becomes 8388608.0 which is wrong.
        ## We need to use the new bsf because we want to use the lowest value
        ## ever and we want beta <= 1.
        assert np.all(self.F_new_bsf != 0), f"F_bsf = {self.F_new_bsf}"
        assert np.all(self.F_new_bsf != 0), f"F_bsf = {self.F_new_bsf}"
        # We use abs to ignore negative numbers. 
        beta = (abs(self.F_new_bsf) + Metrics.eps) / (np.abs(self.F_children) + Metrics.eps)
        # If we do have negative numbers, we may get values >= 1, just reverse.
        beta = np.where(beta > 1, 1 / beta, beta)
        assert ((beta > 0) & (beta <= 1)).all(),f"beta={beta}, F_bsf = {self.F_new_bsf}, F_children = {self.F_children.abs() + Metrics.eps}"
        return self.improv_wrt_parent() * beta
   
class AOS(object):
    __version__ = '1.0'
    
    # FIXME: can we get these names by listing the methods of the Metric class?
    # FIXME: We should move these to Metrics
    OM_choices = {"absolute_fitness": 0, # "offsp_fitness"
                  "exp_absolute_fitness": 1,
                  "improv_wrt_parent": 2,
                  "improv_wrt_pop": 3,
                  "improv_wrt_bsf": 4,
                  "improv_wrt_median": 5,
                  "relative_fitness_improv": 6}

    param_choice = "OM_choice"
    param_choice_help = "Offspring metric selected"

    known_AOS = { #"AP" : {
        #"OM_choice" : [6],
        #"rew_choice" : ["Normalised_success_sum_window"], 
        #"qual_choice": [0],
        #"prob_choice":        [0], 
        #"select_choice" : [1], 
        #"window_size": [20, 150], 
        #"normal_factor" :        [0, 1], 
        #"decay_rate" : [0.0, 1.0], 
        #"p_min" : [0.0, 0.5], 
        #"error_prob" :        [0.0, 1.0], 
        #},
        "ADOPP" : {
            "OM_choice" : "improv_wrt_median",
            "rew_choice" : "Ancestor_success",
            "decay": ([0.0, 1.0], 0.8), # default 0.8
            "max_gen" : ([1, 50], 5), # default 5
            "window_size": ([1, 500], 100), # qlen = 100
            "frac": 0.0,
            "periodic" : False,
            "convergence_factor" : 0, 
            "qual_choice" : "Quality_Identity",
            "prob_choice" : "Probability_Matching",
            "select_choice" : "Proportional_Selection",
            # FIXME: If the ranges don't change, do we need to provide them here?
            "p_min" : 0.0,
            "error_prob" : 0.0,
        },
        
        "ADOPP_ext" : {
            "OM_choice" : ["improv_wrt_median"],
            "rew_choice" : ["Ancestor_success"],
            "decay": [0.0, 1.0], # default 0.8
            "max_gen" : [1, 50],
            "window_size": [1, 500], # qlen = 100
            "frac": [0.01, 0.1], # default 0.01
            "convergence_factor" : [1, 100], # default 20
            "qual_choice" : ["Quality_Identity"],
            "prob_choice" : ["Probability_Matching"],
            "select_choice" : ["Proportional_Selection"],
            "p_min" : [0.0],
            "error_prob" : [0.0],
        },
        "MA_S2" : {
            "OM_choice" : ["relative_fitness_improv"],
            "rew_choice" : ["Total_avg_gen"],
            "max_gen" : [1],
            "qual_choice" : ["Accumulate"],
            "prob_choice" : ["Probability_Matching"],
            "p_min" : [0.0],
            "error_prob" : [0.0],
            "select_choice" : ["Proportional_Selection"],
        },

        "AUC_AP" : {
            "OM_choice" : ["improv_wrt_parent"],
            "rew_choice" : ["Area_Under_The_Curve"],
            "qual_choice" : ["Weighted_sum"],
            "prob_choice" : ["Adaptive_Pursuit"],
            "select_choice" : ["Proportional_Selection"],
            "window_size" : [20, 150],
            "decay" : [0.0, 1.0],
            "decay_rate" : [0.01, 1.0],
            "p_min" : [0.0, 0.5],
            "p_max" : [0.0, 1.0],
            "learning_rate" : [0.0, 1.0],
        },

        "AUC_MAB" : {
            "OM_choice" : ["improv_wrt_parent"],
            "rew_choice" : ["Area_Under_The_Curve"],
            "qual_choice" : ["Upper_confidence_bound"],
            "prob_choice" : ["Probability_Matching"],
            "select_choice" : ["Greedy_Selection"],
            "window_size" : [20, 150],
            "decay" : [0.0, 1.0],
            "scaling_factor" : [0.0, 1.0],
            "p_min" : [0.0, 0.5],
            "error_prob" : [0.0, 1.0],
        },

        "AUC_PM" : {
            "OM_choice" : ["improv_wrt_parent"],
            "rew_choice" : ["Area_Under_The_Curve"],
            "qual_choice" : ["Weighted_sum"],
            "prob_choice" : ["Probability_Matching"],
            "select_choice" : ["Proportional_Selection"],
            "window_size" : [20, 150],
            "decay" : [0.0, 1.0],
            "decay_rate" : [0.01, 1.0],
            "p_min" : [0.0, 0.5],
            "error_prob" : [0.0, 1.0],
        },

        "Compass" : {
            "OM_choice" : ["improv_wrt_parent"],
            "rew_choice" : ["Compass_projection"],
            "qual_choice" : ["Quality_Identity"],
            "prob_choice" : ["Probability_Matching"],
            "select_choice" : ["Proportional_Selection"],
            "fix_appl" : [10, 150],
            "theta" : [36, 45, 54, 90],
            "p_min" : [0.0, 0.5],
            "error_prob" : [0.0, 1.0],
        },

        "Dyn_GEPv1" : {
            "OM_choice" : ["improv_wrt_parent"],
            "rew_choice" : ["Success_sum"],
            "qual_choice" : ["Bellman_Equation"],
            "prob_choice" : ["Probability_Matching"],
            "select_choice" : ["Proportional_Selection"],
            "max_gen" : [1, 50],
            "weight_reward" : [0.0, 1.0],
            "weight_old_reward" : [0.0, 1.0],
            "discount_rate" : [0.0, 1.0],
            "p_min" : [0.0, 0.5],
            "error_prob" : [0.0, 1.0],
        },

        "Dyn_GEPv2" : {
            "OM_choice" : ["improv_wrt_parent"],
            "rew_choice" : ["Normalised_best_sum"],
            "qual_choice" : ["Bellman_Equation"],
            "prob_choice" : ["Probability_Matching"],
            "select_choice" : ["Proportional_Selection"],
            "max_gen" : [1, 50],
            "intensity" : [1, 2, 3],
            "alpha" : [0, 1],
            "weight_reward" : [0.0, 1.0],
            "weight_old_reward" : [0.0, 1.0],
            "discount_rate" : [0.0, 1.0],
            "p_min" : [0.0, 0.5],
            "error_prob" : [0.0, 1.0],
        },

        "Ext_AP" : {
            "OM_choice" : ["improv_wrt_parent"],
            "rew_choice" : ["Normalised_best_sum"],
            "qual_choice" : ["Weighted_sum"],
            "prob_choice" : ["Adaptive_Pursuit"],
            "select_choice" : ["Proportional_Selection"],
            "max_gen" : [1, 50],
            "intensity" : [1, 2, 3],
            "alpha" : [0, 1],
            "decay_rate" : [0.01, 1.0],
            "p_min" : [0.0, 0.5],
            "p_max" : [0.0, 1.0],
            "learning_rate" : [0.0, 1.0],
        },

        "Ext_MAB" : {
            "OM_choice" : ["improv_wrt_parent"],
            "rew_choice" : ["Normalised_best_sum"],
            "qual_choice" : ["Upper_confidence_bound"],
            "prob_choice" : ["Probability_Matching"],
            "select_choice" : ["Greedy_Selection"],
            "max_gen" : [1, 50],
            "intensity" : [1, 2, 3],
            "alpha" : [0, 1],
            "scaling_factor" : [0.0, 1.0],
            "p_min" : [0.0, 0.5],
            "error_prob" : [0.0, 1.0],
        },

        "Ext_PM" : {
            "OM_choice" : ["improv_wrt_parent"],
            "rew_choice" : ["Normalised_best_sum"],
            "qual_choice" : ["Weighted_sum"],
            "prob_choice" : ["Probability_Matching"],
            "select_choice" : ["Proportional_Selection"],
            "max_gen" : [1, 50],
            "intensity" : [1, 2, 3],
            "alpha" : [0, 1],
            "decay_rate" : [0.01, 1.0],
            "p_min" : [0.0, 0.5],
            "error_prob" : [0.0, 1.0],
        },
        #  F. G. Lobo and D. E. Goldberg, “Decision making in a hybrid genetic
        #  algorithm,” in Proceedings of the 1997 IEEE International Conference
        #  on Evolutionary Computation (ICEC’97), T. Bäck, Z. Michalewicz, and
        #  X. Yao, Eds. Piscataway, NJ: IEEE Press, 1997, pp. 121–125.
        "Hybrid" : {
            "OM_choice" : ["improv_wrt_pop"],
            "rew_choice" : ["Best2gen"], # FIXME: ExtMetric
            "max_gen" : [1], 
            #"qual_choice" : ["Bellman_Equation"],
            "qual_choice" : ["Weighted_sum"],
            "q_min" : [0.0], # default
            "discount_rate" : [0.0, 1.0],
            "prob_choice" : ["Probability_Matching"],
            "p_min" : [0.0], # no default given.
            "error_prob" : [0.0],
            "select_choice" : ["Proportional_Selection"],
        },
        # 
        # "Hybridv2" : {
        #     "OM_choice" : ["improv_wrt_parent"],
        #     "rew_choice" : ["Best2gen"],
        #     "qual_choice" : ["Bellman_Equation"],
        #     "prob_choice" : ["Probability_Matching"],
        #     "select_choice" : ["Proportional_Selection"],
        #     "scaling_constant" : [0.001, 1.0],
        #     "alpha" : [0, 1],
        #     "beta" : [0,1],
        #     "weight_reward" : [0.0, 1.0],
        #     "weight_old_reward" : [0.0, 1.0],
        #     "discount_rate" : [0.0, 1.0],
        #     "p_min" : [0.0, 0.5],
        #     "error_prob" : [0.0, 1.0],
        # },

        "MEANS" : {
            "OM_choice" : ["absolute_fitness"],
            "rew_choice" : ["Success_Rate_old"],
            "qual_choice" : ["Upper_confidence_bound"],
            "prob_choice" : ["Probability_Matching"],
            "select_choice" : ["Proportional_Selection"],
            "max_gen" : [1, 50],
            "succ_lin_quad": [1, 2],
            "frac": [0.0, 1.0],
            "noise": [0.0,1.0],
            "scaling_factor" : [0.0, 1.0],
            "p_min" : [0.0, 0.5],
            "error_prob" : [0.0, 1.0],
        },

        "MMRDE" : {
            "OM_choice" : ["improv_wrt_parent"],
            "rew_choice" : ["Success_Rate_old"],
            "qual_choice" : ["Quality_Identity"],
            "prob_choice" : ["Probability_Matching"],
            "select_choice" : ["Proportional_Selection"],
            "max_gen" : [1, 50],
            "succ_lin_quad": [1, 2],
            "frac": [0.0, 1.0],
            "noise": [0.0,1.0],
            "p_min" : [0.0, 0.5],
            "error_prob" : [0.0, 1.0],
        },

        # C. Igel and M. Kreutz, “Using fitness distributions to improve the
        # evolution of learning structures,” in Proceedings of the 1999
        # Congress on Evolutionary Computation (CEC 1999), vol. 3. Piscataway,
        # NJ: IEEE Press, 1999, pp. 1902–1909.
        "Op_adapt" : {
            "OM_choice" : ["improv_wrt_pop"],
            "rew_choice" : ["Total_avg_gen"],
            "periodic" : [1],
            "max_gen" : [1, 50], # default 10
            "qual_choice" : ["Weighted_sum"],
            "decay_rate" : [0.01, 1], # default: 0.5 
            "q_min" : [0.01, 1], # default 0.1
            "prob_choice" : ["Probability_Matching"],
            "p_min" : [0.0],
            "error_prob" : [0.0],
            "select_choice" : ["Proportional_Selection"],
        },

        "Adapt_NN" : {
            "OM_choice" : ["improv_wrt_parent"],
            "rew_choice" : ["Normalized_avg_period"],
            "periodic" : [1],
            "max_gen" : [1, 50], # default 4
            "qual_choice" : ["Weighted_sum"],
            "decay_rate": [0.01, 1.0], # delta default: 0.3
            "q_min" : [1], # 
            "prob_choice" : ["Probability_Matching"],
            "p_min" : [0.01, 1.0], # default pmin = 0.1
            "error_prob" : [0.0],
            "select_choice" : ["Proportional_Selection"],
        },

        
        "PD_PM" : {
            "OM_choice" : ["improv_wrt_parent"],
            "rew_choice" : ["Pareto_Dominance"],
            "qual_choice" : ["Weighted_sum"],
            "prob_choice" : ["Probability_Matching"],
            "select_choice" : ["Proportional_Selection"],
            "fix_appl" : [10, 150],
            "decay_rate" : [0.01, 1.0],
            "p_min" : [0.0, 0.5],
            "error_prob" : [0.0, 1.0],
        },

        "PDP" : {
            "OM_choice" : ["improv_wrt_parent"],
            "rew_choice" : ["Success_rate"],
            "max_gen" : [100000], # default is np.inf
            "gamma": [2],
            "qual_choice" : ["Quality_Identity"],
            "prob_choice" : ["Probability_Matching"],
            "p_min" : [0.0, 0.5], # default is 0.2 / n_ops
            "error_prob" : [0.0],
            "select_choice" : ["Proportional_Selection"],
        },

        "PR_PM" : {
            "OM_choice" : ["improv_wrt_parent"],
            "rew_choice" : ["Pareto_Rank"],
            "qual_choice" : ["Weighted_sum"],
            "prob_choice" : ["Probability_Matching"],
            "select_choice" : ["Proportional_Selection"],
            "fix_appl" : [10, 150],
            "decay_rate" : [0.01, 1.0],
            "p_min" : [0.0, 0.5],
            "error_prob" : [0.0, 1.0],
        },

        "Proj_PM" : {
            "OM_choice" : ["improv_wrt_parent"],
            "rew_choice" : ["Compass_projection"],
            "qual_choice" : ["Weighted_sum"],
            "prob_choice" : ["Probability_Matching"],
            "select_choice" : ["Proportional_Selection"],
            "fix_appl" : [10, 150],
            "theta" : [36, 45, 54, 90],
            "decay_rate" : [0.01, 1.0],
            "p_min" : [0.0, 0.5],
            "error_prob" : [0.0, 1.0],
        },

        "RFI_AA_PM" : {
            "OM_choice" : ["relative_fitness_improv"],
            "rew_choice" : ["Normalised_success_sum_window"],
            "qual_choice" : ["Weighted_sum"],
            "prob_choice" : ["Probability_Matching"],
            "select_choice" : ["Proportional_Selection"],
            "window_size":[20, 150],
            "normal_factor":[0, 1],
            "decay_rate" : [0.01, 1.0],
            "p_min" : [0.0, 0.5],
            "error_prob" : [0.0, 1.0],
        },

        "RFI_EA_PM" : {
            "OM_choice" : ["relative_fitness_improv"],
            "rew_choice" : ["Normalised_best_sum"],
            "qual_choice" : ["Weighted_sum"],
            "prob_choice" : ["Probability_Matching"],
            "select_choice" : ["Proportional_Selection"],
            "max_gen" : [1, 50],
            "intensity" : [1, 2, 3],
            "alpha" : [0, 1],
            "decay_rate" : [0.01, 1.0],
            "p_min" : [0.0, 0.5],
            "error_prob" : [0.0, 1.0],
        },

        "RecPM" : {
            "OM_choice" : ["improv_wrt_parent"],
            "rew_choice" : ["Immediate_Success"],
            "qual_choice" : ["Bellman_Equation"],
            "prob_choice" : ["Probability_Matching"],
            "select_choice" : ["Proportional_Selection"],
            "weight_reward" : [0.0, 1.0],
            "weight_old_reward" : [0.0, 1.0],
            "discount_rate" : [0.0, 1.0],
            "p_min" : [0.0, 0.5],
            "error_prob" : [0.0, 1.0],
        },
            
        "SR_AP" : {
            "OM_choice" : ["improv_wrt_parent"],
            "rew_choice" : ["Sum_of_Rank"],
            "qual_choice" : ["Weighted_sum"],
            "prob_choice" : ["Adaptive_Pursuit"],
            "select_choice" : ["Proportional_Selection"],
            "window_size" : [20, 150],
            "decay" : [0.0, 1.0],
            "decay_rate" : [0.01, 1.0],
            "p_min" : [0.0, 0.5],
            "p_max" : [0.0, 1.0],
            "learning_rate" : [0.0, 1.0],
        },

        "SR_MAB" : {
            "OM_choice" : ["improv_wrt_parent"],
            "rew_choice" : ["Sum_of_Rank"],
            "qual_choice" : ["Upper_confidence_bound"],
            "prob_choice" : ["Probability_Matching"],
            "select_choice" : ["Greedy_Selection"],
            "window_size" : [20, 150],
            "decay" : [0.0, 1.0],
            "scaling_factor" : [0.0, 1.0],
            "p_min" : [0.0, 0.5],
            "error_prob" : [0.0, 1.0],
        },

        "SR_PM" : {
            "OM_choice" : ["improv_wrt_parent"],
            "rew_choice" : ["Sum_of_Rank"],
            "qual_choice" : ["Weighted_sum"],
            "prob_choice" : ["Probability_Matching"],
            "select_choice" : ["Proportional_Selection"],
            "window_size" : [20, 150],
            "decay" : [0.0, 1.0],
            "decay_rate" : [0.01, 1.0],
            "p_min" : [0.0, 0.5],
            "error_prob" : [0.0, 1.0],
        },

        "SaDE" : {
            "OM_choice" : ["improv_wrt_parent"],
            "rew_choice" : ["Success_Rate_old"],
            "qual_choice" : ["Quality_Identity"],
            "prob_choice" : ["Probability_Matching"],
            "select_choice" : ["Proportional_Selection"],
            "max_gen" : [1, 50],
            "succ_lin_quad": [1, 2],
            "frac": [0.0, 1.0],
            "noise": [0.0,1.0],
            "p_min" : [0.0, 0.5],
            "error_prob" : [0.0, 1.0],
        }
    }

    @classmethod
    def build_known_AOS(cls, name,
                        popsize, budget, n_ops, rew_args,
                        qual_args, prob_args, select_args):

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
                    
        print(f'{name} = {dict(OM_choice = OM_choice, rew_choice = rew_choice, rew_args=rew_args, qual_choice=qual_choice, qual_args=qual_args, prob_choice = prob_choice, prob_args=prob_args, select_choice = select_choice, select_args=select_args)}')

        return AOS(popsize, budget, n_ops,
                   OM_choice, rew_choice, rew_args,
                   qual_choice, qual_args,
                   prob_choice, prob_args, select_choice, select_args)
        
        
    def __init__(self, popsize, budget, n_ops, OM_choice, rew_choice, rew_args,
                 qual_choice, qual_args, prob_choice, prob_args, select_choice, select_args):

        self.n_ops = n_ops
        self.metrics = Metrics(minimize=True)
        self.window = OpWindow(n_ops, metric = AOS.OM_choices[OM_choice])
        self.gen_window = GenWindow(n_ops, metric = AOS.OM_choices[OM_choice])
        self.tran_matrix = np.random.rand(n_ops, n_ops)
        self.tran_matrix = normalize_matrix(self.tran_matrix)
        self.probability = np.full(n_ops, 1.0 / n_ops)
        self.old_reward = np.zeros(n_ops)
        self.periodic = rew_args["periodic"] if rew_args["max_gen"] else False
            
        rew_args["popsize"] = popsize
        select_args["popsize"] = popsize
        self.reward_type = build_reward(self, rew_choice, rew_args)
        self.quality_type = build_quality(qual_choice, n_ops, qual_args)
        self.probability_type = build_probability(prob_choice, n_ops, prob_args)
        self.selection_type = build_selection(select_choice, n_ops, select_args, budget)
        self.select_counter = 0
    
    @classmethod
    def add_argument(cls, parser):
        metrics_names = list(cls.OM_choices)
        parser.add_argument("--" + cls.param_choice, choices=metrics_names,
                            help=cls.param_choice_help)
        # Handle rewards
        rew_args_names = RewardType.add_argument(parser)
        # Handle qualities
        qual_args_names = QualityType.add_argument(parser)
        # Handle probabilities
        prob_args_names = ProbabilityType.add_argument(parser)
        # Handle Selection
        select_args_names = SelectionType.add_argument(parser)
        return (rew_args_names, qual_args_names, prob_args_names, select_args_names)
        
    @classmethod
    def irace_parameters(cls, override = {}):
        output = "# " + cls.__name__ + "\n"
        metrics_names = list(cls.OM_choices)
        #choices = range(1, 1 + len(metrics_names))
        if cls.param_choice in override:
            #choices = override[cls.param_choice]
            metrics_names = override[cls.param_choice]
            #choices_help = ', '.join(f"{i}:{j}" for i,j in zip(choices, metrics_names))
        output += irace_parameter(cls.param_choice, object, metrics_names,
                                        help=cls.param_choice_help)
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
                    choices = list(AOS.OM_choices)
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
        return self.selection_type.perform_selection(self.probability, self.select_counter)

############################Offspring Metric definitions#######################
    def OM_Update(self, F, F1, F_bsf, opu):
        """F: fitness of parent population
        F1: fitness of children population
        F_bsf : best so far fitness
        opu: represents (op)erator that produced offspring (u).
        """
        self.metrics.update(F1, F, F_bsf)
        
        #verylarge = 1e32
        
        # See OpWindow metrics
        # Fitness is minimised but metric is maximised
        ## MANUEL: we need to think if this is the best solution to convert to maximization
        ## MUDITA: can't think of anything better than following.
        #absolute_fitness = verylarge - F1
        #assert np.all(absolute_fitness >= 0)
        absolute_fitness = self.metrics.absolute_fitness()
        exp_absolute_fitness = self.metrics.exp_absolute_fitness()
        improv_wrt_parent = self.metrics.improv_wrt_parent()
        improv_wrt_pop = self.metrics.improv_wrt_pop()
        improv_wrt_bsf = self.metrics.improv_wrt_bsf()
        improv_wrt_median = self.metrics.improv_wrt_median()
        # MUDITA: if F_bsf = 1 and F1 = 0, then F_bsf / (F1 + eps) becomes 8388608.0 which is wrong.
        relative_fitness_improv = self.metrics.relative_fitness_improv()
        #relative_fitness_improv = (verylarge - (F_bsf / (F1 + eps))) * improv_wrt_parent
        
        popsize = len(F)
        window_op = np.copy(opu)
        window_met = np.full((popsize, 7), np.nan)
        
        for i in range(popsize):
            # if child is worse or equal to parent, don't store an OM_metric,
            if F1[i] >= F[i]:
                continue
            # MANUEL: If child is worse than parent, we don't store the op, so it is not counted as an application, is that right? It seems wrong.
            # MUDITA: Ofcourse, we need to store the op. This is useful to count the number of unsuccessful applications (Line 158). I have added a line above continue (see two lines above).
            window_met[i, 0] = absolute_fitness[i]
            window_met[i, 1] = exp_absolute_fitness[i]
            window_met[i, 2] = improv_wrt_parent[i]
            if improv_wrt_pop[i] >= 0:
                window_met[i, 3] = improv_wrt_pop[i]
            if improv_wrt_bsf[i] >= 0:
                window_met[i, 4] = improv_wrt_bsf[i]
            if improv_wrt_median[i] >= 0:
                window_met[i, 5] = improv_wrt_median[i]
            window_met[i, 6] = relative_fitness_improv[i]
            assert window_met[i, 2] >= 0
            #assert window_met[i,6] >= 0
            if self.window.max_size:
                self.window.append(window_op[i], window_met[i, :])
        
        self.gen_window.append(window_op, window_met)
        # Update if we are not doing periodic updates or max_gen == 0 or
        # or counter % max_gen == 0
        if not self.periodic or self.gen_window.max_gen == 0 \
           or self.select_counter % self.gen_window.max_gen:
            # MANUEL: Why do we need to return num_op?
            reward, num_op = self.reward_type.calc_reward()
            #old_reward = self.reward_type.old_reward
            #old_prob = self.probability_type.old_probability
            quality = self.quality_type.calc_quality(self.old_reward, reward, num_op, self.tran_matrix)
            self.probability = self.probability_type.calc_probability(quality)
            # FIXME: Why do we need to calculate this here?
            self.tran_matrix = transitive_matrix(self.probability)
            self.old_reward = np.copy(reward)
        # FIXME: This is not really the selection counter, only the update counter.
        self.select_counter += 1

    
###################Other definitions############################################

def transitive_matrix(p):
    """Calculates Transitive Matrix."""
    ## Numpy broadcasting.
    tran_matrix = p + p[:, np.newaxis]
    return normalize_matrix(tran_matrix)

def normalize_matrix(x):
    """Normalise n_ops dimensional matrix""" 
    return x / np.sum(x, axis=1)[:, None]

def calc_delta_r(decay, W, window_size, ndcg):
    if decay == 0:
        return np.ones(window_size)
    r = np.arange(float(W))
    if ndcg:
        r += 1
        delta_r = ((2 ** (W - r)) - 1) / np.log(1 + r)
    else:
        delta_r = (decay ** r) * (W - r)
    return delta_r

def AUC(operators, rank, op, decay, window_size, ndcg = True):
    """Calculates area under the curve for each operator"""
    assert len(operators) == len(rank)
    W = len(operators)
    delta_r_vector = calc_delta_r(decay, W, window_size, ndcg)
    x, y, area = 0, 0, 0
    r = 0
    while r < W:
        delta_r = delta_r_vector[r]
        # number of rewards equal to reward ranked r given by op
        tiesY = np.count_nonzero(rank[operators == op] == rank[r])
        # number of rewards equal to reward ranked r given by others
        tiesX = np.count_nonzero(rank[operators != op] == rank[r])
        assert tiesY >= 0
        assert tiesX >= 0
        if (tiesX + tiesY) > 0 :
            delta_r = np.sum(delta_r_vector[r : r + tiesX + tiesY]) / (tiesX + tiesY)
            x += tiesX * delta_r
            area += (y * tiesX * delta_r) + (0.5 * delta_r * delta_r * tiesX * tiesY)
            y += tiesY * delta_r
            r += tiesX + tiesY
        elif operators[r] == op:
            y += delta_r
            r += 1
        else:
            x += delta_r
            area += y * delta_r
            r += 1
    return area

def UCB(N, C, reward):
    '''Calculates Upper Confidence Bound as a quality'''
    ucb = reward + C * np.sqrt(np.log(np.sum(N)) / N)
    ucb[np.isinf(ucb) | np.isnan(ucb)] = 0.0
    return ucb

##################################################Reward definitions######################################################################
    


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
    def add_argument(cls, parser):
        "Add arguments to an ArgumentParser"
        return parser_add_arguments(cls, parser)

    @classmethod
    def irace_parameters(cls, override = {}):
        return aos_irace_parameters(cls, override = override)

        
class RewardType(BasicType):
    # Static variables
    # FIXME: Use __slots__ to find which parameters need to be defined.
    # FIXME: define this in the class as @property getter doctstring and get it from it
    # MUDITA_check: To generate existiong parameter files, I had to change categorical parameter (theta, succ_lin_quad, normal_factor, alpha, beta, intensity) type to object. Because in irace_parameter() function, categorical is represented as object. But to run this code for parameter tuning, I had to change these categorical to int.
    params = [
        "periodic",       object,      0,     [0, 1], "Periodic update (only every max_gen)",
        "max_gen",          int,        10,     [1, 50],                        "Maximum number of generations for generational window",
        "gamma",           object,       1,     [1, 2, 3],                        "Exponent (linear, quadratic, ...) of Success_rate",
        "fix_appl",         int,        20,     [10, 150],                      "Maximum number of successful operator applications for generational window",
        "theta",            object,        45,     [36, 45, 54, 90],               "Search direction",
        "window_size",      int,        50,     [1, 500],                      "Size of window",
        "decay",            float,      0.4,    [0.0, 1.0],                     "Decay value to emphasise the choice of better operator",
        "succ_lin_quad",    object,        1,      [1, 2],                         "Operator success as linear or quadratic",
        "frac",             float,      0.01,   [0.0, 0.1],                     "Fraction of sum of successes of all operators",
        "noise",            float,      0.0,    [0.0, 1.0],                     "Small noise for randomness",
        "normal_factor",    object,        1,      [0, 1],                         "Choice to normalise",
        "scaling_constant", float,      1,      [0.001, 1.0],                   "Scaling constant (C)",
        "alpha",            object,        0,      [0, 1],                         "Choice to normalise by best produced by any operator",
        "beta",             object,        1,      [0, 1],                         "Choice to include the difference between budget used by an operator in previous two generations",
        "intensity",        object,        1,      [1, 2, 3],                      "Intensify the changes of best fitness value",
        "convergence_factor",    float, 20,        [0, 100],    "Factor for convergence credits",
    ]
    params_conditions = {
        "max_gen": ["Success_rate", "Success_Rate_old", "Success_sum", "Total_avg_gen", "Normalised_best_sum", "Ancestor_success", "Normalized_avg_period"],
        "fix_appl": ["Pareto_Dominance", "Pareto_Rank", "Compass_projection"],
        "theta": ["Compass_projection"],
        "window_size": ["Area_Under_The_Curve", "Sum_of_Rank", "Normalised_success_sum_window", "Ancestor_success"],
        "decay": ["Area_Under_The_Curve", "Sum_of_Rank", "Ancestor_success"],
        "gamma": ["Success_rate"],
        "succ_lin_quad" : ["Success_Rate_old"],
        "frac": ["Ancestor_success","Success_Rate_old"],
        "noise": ["Success_Rate_old"],
        "normal_factor": ["Normalised_success_sum_window"],
        "scaling_constant": ["Best2gen"],
        "alpha" : ["Best2gen", "Normalised_best_sum"],
        "beta": ["Best2gen"],
        "intensity": ["Normalised_best_sum"],
        "convergence_factor" : ["Ancestor_success"]
    }
    # param_choices = [
    #     "Pareto_Dominance",
    #     "Pareto_Rank",
    #     "Compass_projection",
    #     "Area_Under_The_Curve",
    #     "Sum_of_Rank",
    #     "Success_Rate_old",
    #     "Immediate_Success",
    #     "Success_sum",
    #     "Normalised_success_sum_window",
    #     "Total_avg_gen",
    #     "Best2gen",
    #     "Normalised_best_sum"
    # ]
    param_choice = "rew_choice"
    param_choice_help = "Reward method selected"

    def __init__(self, aos, max_gen = None, window_size = None, decay = None, fix_appl = None):
        # Set a few common short-cuts.
        self.n_ops = aos.n_ops
        self.metrics = aos.metrics
        if max_gen:
            self.gen_window = aos.gen_window
            self.gen_window.max_gen = max_gen
        if window_size:
            self.window = aos.window
            self.window.resize(window_size)
            self.window_size = window_size
        self.decay = decay
        self.fix_appl = fix_appl
        self.eps = np.finfo(np.float32).eps
        

    def check_reward(self, reward, num_op):
        # FIXME: What is num_op?
        # MANUEL: Can reward be negative?
        # MUDITA: Relative_fitness_improv holds negtaive values which might lead to negative reward value.
        rew_min = reward.min()
        rew_diff = reward.max() - rew_min
        if rew_diff > 0:
            reward = (reward - rew_min) / rew_diff
        else:
            reward[:] = 0.0
            
        assert np.all(np.isfinite(reward)), f"Infinite reward {reward}"
        assert np.all(reward >= 0), f"Negative reward {reward}"
        #debug_print("{:>30}:      reward={}".format(type(self).__name__, reward))
        return reward, num_op

    @abstractmethod
    def calc_reward(self):
        pass

class Pareto_Dominance(RewardType):
    """
Jorge Maturana, Fr ́ed ́eric Lardeux, and Frederic Saubion. “Autonomousoperator management for evolutionary algorithms”. In:Journal of Heuris-tics16.6 (2010).https://link.springer.com/content/pdf/10.1007/s10732-010-9125-3.pdf, pp. 881–909.
"""

    def __init__(self, aos, fix_appl = 20):
        # This uses the gen_window, shouldn't we limit the size of the window?
        super().__init__(aos, fix_appl = fix_appl)
        #debug_print("{:>30}: fix_appl = {}".format(type(self).__name__, self.fix_appl))
    
    def calc_reward(self):
        # Pareto dominance returns the number of operators dominated by an
        # operator whereas Pareto rank gives the number of operators an
        # operator is dominated by.
        N = np.zeros(self.n_ops)
        std_op = np.full(self.n_ops, np.nan)
        mean_op = np.full(self.n_ops, np.nan)
        for i in range(self.n_ops):
            b = self.gen_window.metric_for_fix_appl_of_op(i, self.fix_appl)
            N[i] = len(b)
            if N[i] > 0:
                std_op[i] = np.std(b)
                mean_op[i] = np.mean(b)

        reward = np.zeros(self.n_ops)
        for i in range(self.n_ops):
            if np.isnan(std_op[i]):
                continue
            for j in range(self.n_ops):
                if i == j or np.isnan(std_op[j]):
                    continue
                # We want to minimize the std but maximise the mean quality.
                # Count if j dominates i.
                if std_op[i] < std_op[j] and mean_op[i] > mean_op[j]:
                    reward[i] += 1
        reward = normalize_sum(reward)
        return super().check_reward(reward, N)


class Pareto_Rank(RewardType):
    """
Jorge Maturana, Fr ́ed ́eric Lardeux, and Frederic Saubion. “Autonomous operator management for evolutionary algorithms”. In:Journal of Heuris-tics16.6 (2010).https://link.springer.com/content/pdf/10.1007/s10732-010-9125-3.pdf, pp. 881–909.
"""
    def __init__(self, aos, fix_appl = 20):
        super().__init__(aos, fix_appl = fix_appl)
        #debug_print("{:>30}: fix_appl = {}".format(type(self).__name__, self.fix_appl))

    def calc_reward(self):
        N = np.zeros(self.n_ops)
        std_op = np.full(self.n_ops, np.nan)
        mean_op = np.full(self.n_ops, np.nan)
        for i in range(self.n_ops):
            b = self.gen_window.metric_for_fix_appl_of_op(i, self.fix_appl)
            N[i] = len(b)
            if N[i] > 0:
                std_op[i] = np.std(b)
                mean_op[i] = np.mean(b)

        reward = np.zeros(self.n_ops)
        for i in range(self.n_ops):
            if np.isnan(std_op[i]):
                continue
            for j in range(self.n_ops):
                if i == j or np.isnan(std_op[j]):
                    continue
                # We want to minimize the std but maximise the mean quality.
                # Count if j dominates i.
                if std_op[j] < std_op[i] and mean_op[j] > mean_op[i]:
                    reward[i] += 1
        
        reward = normalize_sum(reward)
        reward = 1. - reward
        return super().check_reward(reward, N)


class Compass_projection(RewardType):
    """
        Jorge Maturana and Fr ́ed ́eric Saubion. “A compass to guide genetic al-gorithms”. In:International Conference on Parallel Problem Solving fromNature.http://www.info.univ-angers.fr/pub/maturana/files/MaturanaSaubion-Compass-PPSNX.pdf. Springer. 2008, pp. 256–265.
        """
    def __init__(self, aos, fix_appl = 100, theta = 45):
        super().__init__(aos, fix_appl = fix_appl)
        self.theta = int(theta)
        #debug_print("{:>30}: fix_appl = {}".format(type(self).__name__, self.fix_appl, self.theta))
    
    def calc_reward(self):
        N = np.zeros(self.n_ops)
        reward = np.zeros(self.n_ops)
        std = np.zeros(self.n_ops)
        avg = np.zeros(self.n_ops)
        angle = np.zeros(self.n_ops)
        # Projection on line B with theta = pi/4
        #        B = [1, 1]
        for i in range(self.n_ops):
            b = self.gen_window.metric_for_fix_appl_of_op(i, self.fix_appl)
            N[i] = len(b)
            if N[i] > 0:
                # Diversity
                std[i] = np.std(b)
                # Quality 
                avg[i] = np.mean(b)
        if np.max(std) != 0:
            std = std / np.max(std)
        if np.max(avg) != 0:
            avg = avg / np.max(avg)
        # MANUEL: What should happen if both are zero?
        # MUDITA: Conceptually, its okay to have avg and std as 0. In that case coordinate will be on origin and there won't be any projection. Thus perpendicular distance fom coordinate to plane will be 0. So to deal with 0s, I have added eps in denominator. But again the issue is 1/(0+eps) = 8388608 which is wrong. Conceptually, (std, avg) = (0, 1) is possible. On scientific calulator arctan(1/0) = 1.5707... But in python it gives division by 0 error.
        # assert avg != 0.0 and std != 0.0
        where = np.flatnonzero((std!=0) | (avg!=0))
        angle[where] = np.fabs(np.arctan(np.deg2rad(avg[where] / std[where]) - np.deg2rad(self.theta)))
        # Euclidean distance of the vector
        reward = (np.sqrt(std**2 + avg**2)) * np.cos(angle)
        # Maturana & Sablon (2008) divide by T_it defined as mean
        # execution time of operator i over its last t applications.
        # We do not divide
        reward = reward - np.min(reward)
        return super().check_reward(reward, N)

class Area_Under_The_Curve(RewardType):
    """
Alvaro Fialho, Marc Schoenauer, and Mich`ele Sebag. “Toward comparison-based adaptive operator selection”. In:Proceedings of the 12th annual con-ference on Genetic and evolutionary computation.https://hal.inria.fr/file/index/docid/471264/filename/banditGECCO10.pdf. ACM.2010, pp. 767–774
"""
    def __init__(self, aos, window_size = 50, decay = 0.4):
        super().__init__(aos, window_size = window_size, decay = decay)
        #debug_print("{:>30}: window_size = {}, decay = {}".format(type(self).__name__, self.window_size, self.decay))
    
    def calc_reward(self):
        reward = np.zeros(self.n_ops)
        window = self.window.truncate(self.window_size)
        window_op_sorted, rank = window.get_ops_sorted_and_rank()
        for op in range(self.n_ops):
            reward[op] = AUC(window_op_sorted, rank, op, self.window_size, self.decay)
        N = window.count_ops()
        return super().check_reward(reward, N)

class Sum_of_Rank(RewardType):
    """
Alvaro Fialho, Marc Schoenauer, and Mich`ele Sebag. “Toward comparison-based adaptive operator selection”. In:Proceedings of the 12th annual con-ference on Genetic and evolutionary computation.https://hal.inria.fr/file/index/docid/471264/filename/banditGECCO10.pdf. ACM.2010, pp. 767–774
"""
    def __init__(self, aos, window_size = 50, decay = 0.4):
        super().__init__(aos, window_size = window_size, decay = decay)
        #debug_print("{:>30}: window_size = {}, decay = {}".format(type(self).__name__, self.window_size, self.decay))
    
    def calc_reward(self):
        reward = np.zeros(self.n_ops)
        window = self.window.truncate(self.window_size)
        window_op_sorted, rank = window.get_ops_sorted_and_rank()
        # Fialho's thesis: https://tel.archives-ouvertes.fr/tel-00578431/document (pg. 79).
        value = (self.decay ** rank) * (self.window_size - rank)
        for i in range(self.n_ops):
            reward[i] = value[window_op_sorted == i].sum()
        reward = normalize_sum(reward)
        N = window.count_ops()
        return super().check_reward(reward, N)


class Success_rate(RewardType):
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
        assert gamma >= 1 and gamma <= 4
        # When the window is large, we speed up computation by keeping our own
        # counts. This breaks the sliding window, but for such large windows it
        # doesn't really make sense to worry about that.
        self.use_window = max_gen < 10000
        self.success = np.zeros(self.n_ops)
        self.used = np.ones(self.n_ops)
        #debug_print("{:>30}: max_gen = {}, succ_lin_quad = {}, frac = {}, noise = {}".format(type(self).__name__, self.gen_window.max_gen, self.succ_lin_quad, self.frac, self.noise))
    
    def calc_reward(self):
        N = np.zeros(self.n_ops)
        max_gen = self.gen_window.get_max_gen()
        last_gen = len(self.gen_window) - 1
        # If the window is anyway not full, use pre-computed.
        if self.use_window or max_gen < last_gen:
            self.success, self.used = self.gen_window.total_success()
        else:
            success, used = self.gen_window.success_ratio(last_gen)
            self.success += success
            self.used += used
        N += self.success
        reward = (self.success**gamma) / self.used
        return super().check_reward(reward, N)


class Success_Rate_old(RewardType):
    """ 

A Kai Qin, Vicky Ling Huang, and Ponnuthurai N Suganthan. “Differ-
ential evolution algorithm with strategy adaptation for global numeri-
cal optimization”. In: IEEE transactions on Evolutionary Computation
13.2 (2009). https://www.researchgate.net/profile/Ponnuthurai_
Suganthan/publication/224330344_Differential_Evolution_Algorithm_
With_Strategy_Adaptation_for_Global_Numerical_Optimization/
links/0c960525d39935a20c000000.pdf, pp. 398–417.

With noise == 0, we get

Bryant A Julstrom. “Adaptive operator probabilities in a genetic algo-
rithm that applies three operators”. In: Proceedings of the 1997 ACM
symposium on Applied computing. http://delivery.acm.org/10.1145/
340000/331746/p233-julstrom.pdf?ip=144.32.48.138&id=331746&
acc=ACTIVE%20SERVICE&key=BF07A2EE685417C5%2E26BE4091F5AC6C0A%
2E4D4702B0C3E38B35 % 2E4D4702B0C3E38B35 & _ _ acm _ _ = 1540905461 _
4567820ac9495f6bfbb8462d1c4244a3. ACM. 1997, pp. 233–238.

"""
    
    def __init__(self, aos, max_gen = 10, succ_lin_quad = 1, frac = 0.01, noise = 0.0):
        # Hyper-parameter values are first assigned here, before this point its none.
        super().__init__(aos, max_gen = max_gen)
        self.succ_lin_quad = int(succ_lin_quad)
        self.frac = frac
        self.noise = noise
        #debug_print("{:>30}: max_gen = {}, succ_lin_quad = {}, frac = {}, noise = {}".format(type(self).__name__, self.gen_window.max_gen, self.succ_lin_quad, self.frac, self.noise))
    
    def calc_reward(self):
        N = np.zeros(self.n_ops)
        gen_window_len = len(self.gen_window)
        max_gen = self.gen_window.get_max_gen()
        reward = np.zeros(self.n_ops)
        for j in range(gen_window_len - max_gen, gen_window_len):
            total_success, total_unsuccess = self.gen_window.count_total_succ_unsucc(j)
            N += total_success
            napplications = total_success + total_unsuccess
            # Avoid division by zero. If total == 0, then total_success is zero.
            napplications[napplications == 0] = 1
            reward += (total_success ** self.succ_lin_quad + self.frac * np.sum(total_success)) / napplications
        reward += self.noise
        return super().check_reward(reward, N)

class Immediate_Success(RewardType):
    """
 Mudita  Sharma,  Manuel  L ́opez-Ib ́a ̃nez,  and  Dimitar  Kazakov.  “Perfor-mance Assessment of Recursive Probability Matching for Adaptive Oper-ator Selection in Differential Evolution”. In:International Conference onParallel Problem Solving from Nature.http://eprints.whiterose.ac.uk/135483/1/paper_66_1_.pdf. Springer. 2018, pp. 321–333.
y """
    def __init__(self, aos, popsize):
        # FIXME: This uses gen_window but not max_gen???
        super().__init__(aos)
        self.popsize = popsize
    
    def calc_reward(self):
        gen_window_len = len(self.gen_window)
        total_success, total_unsuccess = self.gen_window.count_total_succ_unsucc(gen_window_len - 1)
        reward = total_success / self.popsize
        return super().check_reward(reward, total_success)

class Normalized_avg_period(RewardType):
    """
C. Igel and M. Kreutz, “Operator adaptation in evolutionary computation and its application to structure optimization of neural networks,” Neurocomputing, vol. 55, no. 1-2, pp. 347–361, 2003.
    """

    def __init__(self, aos, max_gen = 4):
        super().__init__(aos, max_gen = max_gen)
        assert max_gen > 0
        #debug_print("{:>30}: max_gen = {}".format(type(self).__name__, self.gen_window.max_gen))

    def calc_reward(self):
        N = np.zeros(self.n_ops)
        # Most recent generation
        gen_window_len = len(self.gen_window)
        # Oldest generation
        max_gen = self.gen_window.get_max_gen()
        reward = np.zeros(self.n_ops)
        napplications = np.zeros(self.n_ops)
        for j in range(gen_window_len - max_gen, gen_window_len):
            total_success, total_unsuccess = self.gen_window.count_total_succ_unsucc(j)
            N += total_success
            napplications += total_success + total_unsuccess
            value = self.gen_window.sum_at_generation(j)
            reward += value
        # To avoid division by zero (assumes that value == 0 for these)
        napplications[napplications == 0] = 1
        reward /= napplications
        # FIXME: The paper also normalizes by the total but this could be optional or it could be done in the quality component.
        reward = normalize_sum(reward)        
        return super().check_reward(reward, N)

class Success_sum(RewardType):
    """
 Christian  Igel  and  Martin  Kreutz.  “Operator  adaptation  in  evolution-ary  computation  and  its  application  to  structure  optimization  of  neu-ral  networks”.  In:Neurocomputing55.1-2  (2003).https : / / ac . els -cdn.com/S0925231202006288/1-s2.0-S0925231202006288-main.pdf?_tid=c6274e78-02dc-4bf6-8d92-573ce0bed4c4&acdnat=1540907096_d0cc1e2b4ca56a49587b4d55e1008a84, pp. 347–361.
 """
    def __init__(self, aos, max_gen = 4):
        super().__init__(aos, max_gen = max_gen)
        #debug_print("{:>30}: max_gen = {}".format(type(self).__name__, self.gen_window.max_gen))
    
    def calc_reward(self):
        N = np.zeros(self.n_ops)
        gen_window_len = len(self.gen_window)
        max_gen = self.gen_window.get_max_gen()
        napplications = np.zeros(self.n_ops)
        reward = np.zeros(self.n_ops)
        for j in range(gen_window_len - max_gen, gen_window_len):
            total_success, total_unsuccess = self.gen_window.count_total_succ_unsucc(j)
            N += total_success
            napplications += total_success + total_unsuccess
            value = self.gen_window.sum_at_generation(j)
            reward += value
        napplications[napplications == 0] = 1
        reward /= napplications
        return super().check_reward(reward, N)

class Normalised_success_sum_window(RewardType):
    """
Alvaro Fialho, Marc Schoenauer, and Mich`ele Sebag. “Analysis of adaptive operator selection techniques on the royal road and long k-path problems”.In:Proceedings of the 11th Annual conference on Genetic and evolutionarycomputation.https://hal.archives-ouvertes.fr/docs/00/37/74/49/PDF/banditGECCO09.pdf. ACM. 2009, pp. 779–786.
"""
    def __init__(self, aos, window_size = 50, normal_factor = 1):
        super().__init__(aos, window_size = window_size)
        self.normal_factor = int(normal_factor)
        #debug_print("{:>30}: window_size = {}, normal_factor = {}".format(type(self).__name__, self.window_size, self.normal_factor))
    
    def calc_reward(self):
        reward = np.zeros(self.n_ops)
        # Create a local truncated window.
        window = self.window.truncate(self.window_size)
        N = window.count_ops()
        N_copy = np.copy(N)
        N[N == 0] = 1
        reward = window.sum_per_op() / N
        if np.max(reward) != 0:
            reward /= np.max(reward)**self.normal_factor
        return super().check_reward(reward, N_copy)

class Total_avg_gen(RewardType): # This was Normalised_success_sum_gen
    """
C. Igel and M. Kreutz, “Using fitness distributions to improve the evolution of learning structures,” in Proceedings of the 1999 Congress on Evolutionary Computation (CEC 1999), vol. 3. Piscataway, NJ: IEEE Press, 1999, pp. 1902–1909.
"""
    def __init__(self, aos, max_gen = 4):
        super().__init__(aos, max_gen = max_gen)
        assert max_gen > 0
        #debug_print("{:>30}: max_gen = {}".format(type(self).__name__, self.gen_window.max_gen))
    
    def calc_reward(self):
        N = np.zeros(self.n_ops)
        # Most recent generation
        gen_window_len = len(self.gen_window)
        # Oldest generation
        max_gen = self.gen_window.get_max_gen()
        reward = np.zeros(self.n_ops)
        for j in range(gen_window_len - max_gen, gen_window_len):
            # FIXME: If we can remove N, we can have self.gen_window.avg_at_generation(self,gen)
            total_success, total_unsuccess = self.gen_window.count_total_succ_unsucc(j)
            N += total_success
            napplications = total_success + total_unsuccess
            # To avoid division by zero (assumes that value == 0 for these)
            napplications[napplications == 0] = 1
            value = self.gen_window.sum_at_generation(j)
            reward += value / napplications
        return super().check_reward(reward, N)

class Best2gen(RewardType):
    """
Giorgos Karafotias, Agoston Endre Eiben, and Mark Hoogendoorn. “Genericparameter  control  with  reinforcement  learning”.  In:Proceedings of the2014 Annual Conference on Genetic and Evolutionary Computation.http://www.few.vu.nl/~gks290/papers/GECCO2014-RLControl.pdf. ACM.2014, pp. 1319–1326.
 """
    def __init__(self, aos, scaling_constant = 1., alpha = 0, beta = 1):
        super().__init__(aos, max_gen = 2)
        assert scaling_constant > 0. and scaling_constant <= 1.
        self.scaling_constant = scaling_constant
        self.alpha = int(alpha)
        assert self.alpha == 0 or self.alpha == 1
        self.beta = int(beta)
        assert self.beta == 0 or self.beta == 1
        #debug_print("{:>30}: scaling constant = {}, alpha = {}, beta = {}".format(type(self).__name__, self.scaling_constant, self.alpha, self.beta))
    
    def calc_reward(self):
        gen_window_len = len(self.gen_window)
        
        # Calculating best in current generation
        best_g = self.gen_window.max_at_generation(gen_window_len - 1)
        # Calculating best in previous generation
        if gen_window_len >= 2:
            best_g_1 = self.gen_window.max_at_generation(gen_window_len - 2)
        else:
            best_g_1 = np.zeros(self.n_ops)

        if self.alpha == 1:
            denom = best_g_1.copy()
            # We want to avoid division by zero
            denom[denom == 0] = 1.
        else:
            denom = 1.

        total_success_g, total_unsuccess_g = self.gen_window.count_total_succ_unsucc(gen_window_len - 1)
        if gen_window_len >= 2:
            total_success_g_1, total_unsuccess_g_1 = self.gen_window.count_total_succ_unsucc(gen_window_len - 2)
        else:
            total_success_g_1 = total_unsuccess_g_1 = 0
        N = total_success_g + total_success_g_1
        
        if self.beta == 1:
            n_applications = (total_success_g + total_unsuccess_g) - (total_success_g_1 + total_unsuccess_g_1)
            # We want to avoid division by zero
            n_applications[n_applications == 0] = 1
            n_applications = np.fabs(n_applications)
        else:
            n_applications = 1.

        # FIXME why the fabs here? If the value is negative, isn't that bad?
        reward = self.scaling_constant * np.fabs(best_g - best_g_1) / (denom  * n_applications)
        return super().check_reward(reward, N)

class Ancestor_success(RewardType):
    '''
    B. A. Julstrom, "What have you done for me lately? adapting operator probabilities in a steady-state genetic algorithm," in ICGA, L. J. Eshelman, Ed. Morgan Kaufmann Publishers, San Francisco, CA, 1995, pp. 81–87.

    B. A. Julstrom, "An inquiry into the behavior of adaptive operator probabilities in steady-state genetic algorithms,” in Proceedings of the Second Nordic Workshop on Genetic Algorithms and their Applications, August, 1996, pp. 15–26.
    '''
    def __init__(self, aos, window_size = 100, max_gen = 5, decay = 0.8, frac = 0.01, convergence_factor = 20):
        super().__init__(aos, window_size = window_size, max_gen = max_gen)
        assert decay >= 0.0 and decay <= 1.0
        # The most recent generation is max_gen - 1.
        self.decay_g = decay ** np.arange(max_gen - 1, -1, -1)
        assert frac >= 0.0 and frac <= 1.0
        self.frac = frac

        assert convergence_factor >= 0
        self.convergence_factor = convergence_factor
        #debug_print("{:>30}: max_gen = {}".format(type(self).__name__, self.gen_window.max_gen))
        
        # In the paper, we use the application window, however this is closer
        # to a quality queue.  In any case, it is more efficient to memorise
        # the values as a queue
        self.accum_credit = deque(maxlen = window_size)
        self.total_credit = deque(maxlen = window_size)
        
    def calc_reward(self):
        reward = np.zeros(self.n_ops)

        last_gen = len(self.gen_window) - 1
        succ = self.gen_window.is_success(last_gen)
        which_succ = np.where(succ)[0]
        N = np.zeros(self.n_ops)
        max_gen = self.gen_window.get_max_gen()
        for i in which_succ:
            # For each child i, we get the operators applied to it
            ## FIXME: The current implementation of gen_window assumes that all
            ## parents of i are stored at the same place of the population,
            ## which is true for DE but not in general..
            ops = self.gen_window._gen_window_op[-max_gen:, i]
            N[ops[0]] += 1
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
            
        ops_count = self.window.count_ops()
        reward = np.where(ops_count > 0, tot_acc_reward / ops_count, 0.0)
        return super().check_reward(reward, N)

class Normalised_best_sum(RewardType):
    """
Alvaro Fialho, Marc Schoenauer, and Mich`ele Sebag. “Analysis of adaptiveoperator selection techniques on the royal road and long k-path problems”.In:Proceedings of the 11th Annual conference on Genetic and evolutionarycomputation.https://hal.archives-ouvertes.fr/docs/00/37/74/49/PDF/banditGECCO09.pdf. ACM. 2009, pp. 779–786.
"""
    def __init__(self, aos, max_gen = 10, intensity = 1, alpha = 1):
        super().__init__(aos, max_gen = max_gen)
        self.intensity = int(intensity)
        self.alpha = int(alpha)
        #debug_print("{:>30}: max_gen = {}, intensity = {}, alpha = {}".format(type(self).__name__, self.gen_window.max_gen, self.intensity, self.alpha))
    
    def calc_reward(self):
        # Normalised best sum
        reward = np.zeros(self.n_ops)
        max_gen = self.gen_window.get_max_gen()
        for i in range(self.n_ops):
            reward[i] = np.sum(self.gen_window.max_per_generation(i))
        reward = (1.0 / max_gen) * (reward**self.intensity)
        reward[reward == 0.0] = 1.0
        reward = reward / np.max(reward)**self.alpha
        
        N = np.zeros(self.n_ops)
        gen_window_len = len(self.gen_window)
        max_gen = self.gen_window.get_max_gen()
        for j in range(gen_window_len - max_gen, gen_window_len):
            total_success, total_unsuccess = self.gen_window.count_total_succ_unsucc(j)
            N += total_success
        return super().check_reward(reward, N)


##################################################Quality definitions######################################################################

def get_choices_dict(cls):
    return { x.__name__:x for x in cls.__subclasses__()}
    
def build_reward(aos, choice, rew_args):
    # Use a dictionary so that we don't have to retype the name.
    choices = get_choices_dict(RewardType)

    if choice == "Pareto_Dominance":
        return choices[choice](aos = aos, fix_appl = rew_args["fix_appl"])
    elif choice == "Pareto_Rank":
        return choices[choice](aos = aos, fix_appl = rew_args["fix_appl"])
    elif choice == "Compass_projection":
        return choices[choice](aos = aos, fix_appl = rew_args["fix_appl"], theta = rew_args["theta"])
    elif choice == "Area_Under_The_Curve":
        return choices[choice](aos = aos, window_size = rew_args["window_size"], decay = rew_args["decay"])
    elif choice == "Sum_of_Rank":
        return choices[choice](aos = aos, window_size = rew_args["window_size"], decay = rew_args["decay"])
    elif choice == "Success_Rate_old":
        return choices[choice](aos = aos, max_gen = rew_args["max_gen"], succ_lin_quad = rew_args["succ_lin_quad"], frac = rew_args["frac"], noise = rew_args["noise"])
    elif choice == "Success_Rate":
        return choices[choice](aos = aos, max_gen = rew_args["max_gen"], gamma = rew_args["gamma"])
    elif choice == "Ancestor_success":
        # FIXME: This is ugly. There must be a way to just pass rew_args as **kwargs and let the called function sort out the keywords. 
        return choices[choice](aos = aos, window_size = rew_args["window_size"], max_gen = rew_args["max_gen"], decay = rew_args["decay"], frac = rew_args["frac"], convergence_factor = rew_args["convergence_factor"])
    elif choice == "Immediate_Success":
        return choices[choice](aos = aos, popsize = rew_args["popsize"])
    elif choice == "Success_sum":
        return choices[choice](aos = aos, max_gen = rew_args["max_gen"])
    elif choice == "Normalized_avg_period":
        return choices[choice](aos = aos, max_gen = rew_args["max_gen"])
    elif choice == "Normalised_success_sum_window":
        return choices[choice](aos = aos, window_size = rew_args["window_size"], normal_factor = rew_args["normal_factor"])
    elif choice == "Total_avg_gen":
        return choices[choice](aos = aos, max_gen = rew_args["max_gen"])
    elif choice == "Best2gen":
        return choices[choice](aos = aos, scaling_constant = rew_args["scaling_constant"], alpha = rew_args["alpha"], beta = rew_args["beta"])
    elif choice == "Normalised_best_sum":
        return choices[choice](aos = aos, max_gen = rew_args["max_gen"], intensity = rew_args["intensity"], alpha = rew_args["alpha"])
    else:
        raise ValueError(f"reward choice {choice} unknown")

def build_quality(choice, n_ops, qual_args):
    # Use a dictionary so that we don't have to retype the name.
    choices = get_choices_dict(QualityType)

    if choice == "Weighted_sum":
        return choices[choice](n_ops, qual_args["decay_rate"], qual_args["q_min"])
    elif choice == "Accumulate":
        return choices[choice](n_ops)
    elif choice == "Upper_confidence_bound":
        return choices[choice](n_ops, qual_args["scaling_factor"])
    elif choice == "Quality_Identity":
        return choices[choice](n_ops)
    elif choice == "Weighted_normalised_sum":
        return choices[choice](n_ops, qual_args["decay_rate"], qual_args["q_min"])
    elif choice == "Bellman_Equation":
        return choices[choice](n_ops, qual_args["weight_reward"], qual_args["weight_old_reward"], qual_args["discount_rate"])
    else:
        raise ValueError(f"quality choice {choice} unknown")


class QualityType(BasicType):
    # Static variables
    # FIXME: Use __slots__ to find which parameters need to be defined.
    # FIXME: define this in the class as @property getter doctstring and get it from it
    params = [
        "scaling_factor",    float, 0.5,    [0.01, 100],    "Scaling Factor",
        "decay_rate",        float, 0.6,    [0.01, 1.0],     "Decay rate (delta)",
        "q_min",             float, 0.1,    [0.0, 1.0],     "Minimum quality attained by an operator (divided by num operators)",
        "weight_reward",     float, 1,      [0.0, 1.0],     "Memory for current reward",
        "weight_old_reward", float, 0.9,    [0.0, 1.0],     "Memory for previous reward",
        "discount_rate",     float, 0.0,    [0.01, 1.0],    "Discount rate"
    ]
    params_conditions = {
        "scaling_factor" : ["Upper_confidence_bound"],
        "decay_rate": ["Weighted_sum", "Weighted_normalised_sum"],
        "q_min": ["Weighted_sum", "Weighted_normalised_sum"],
        "weight_reward": ["Bellman_Equation"],
        "weight_old_reward": ["Bellman_Equation"],
        "discount_rate": ["Bellman_Equation"]
    }
    param_choice = "qual_choice"
    param_choice_help = "Quality method selected"
    
    def __init__(self, n_ops):
        self.n_ops = n_ops
        self.old_quality = np.zeros(n_ops)
        self.eps = np.finfo(self.old_quality.dtype).eps
        
    def check_quality(self, quality):
        assert np.sum(quality) >= 0
        # FIXME: why do we need to normalize it to sum to 1?
        quality = normalize_sum(quality)
        self.old_quality[:] = quality[:]
        #debug_print("{:>30}:     quality={}".format(type(self).__name__, quality))
        return quality
    
    @abstractmethod
    def calc_quality(self, old_reward, reward, len_var, tran_matrix):
        pass

class Weighted_sum(QualityType):
    """
 Dirk Thierens. “An adaptive pursuit strategy for allocating operator probabilities”.  In:Proceedings of the 7th annual conference on Genetic andevolutionary computation.http://www.cs.bham.ac.uk/~wbl/biblio/gecco2005/docs/p1539.pdf. ACM. 2005, pp. 1539–1546.
 """
    def __init__(self, n_ops, decay_rate = 0.6, q_min = 0.0):
        super().__init__(n_ops)
        assert decay_rate >= 0 and decay_rate <= 1
        self.decay_rate = decay_rate
        self.q_min = q_min / self.n_ops
        #debug_print("{:>30}: decay_rate = {}".format(type(self).__name__, self.decay_rate))
    
    def calc_quality(self, old_reward, reward, len_var, tran_matrix):
        quality = self.decay_rate * np.maximum(self.q_min, reward) + (1.0 - self.decay_rate) * self.old_quality
        return super().check_quality(quality)

class Upper_confidence_bound(QualityType):
    """
Alvaro Fialho et al. “Extreme value based adaptive operator selection”.In:International Conference on Parallel Problem Solving from Nature.https : / / hal . inria . fr / file / index / docid / 287355 / filename /rewardPPSN.pdf. Springer. 2008, pp. 175–184
"""
    def __init__(self, n_ops, scaling_factor = 0.5):
        super().__init__(n_ops)
        self.scaling_factor = scaling_factor
        #debug_print("{:>30}: scaling_factor = {}".format(type(self).__name__, self.scaling_factor))

    # FIXME: What is num_op?
    # FIXME: Is old_reward actually old_quality?
    def calc_quality(self, old_reward, reward, num_op, tran_matrix):
        num_op[num_op == 0] = 1
        quality = UCB(num_op, self.scaling_factor, reward)
        return super().check_quality(quality)

class Accumulate(QualityType):
    def __init__(self, n_ops):
        super().__init__(n_ops)
    
    def calc_quality(self, old_reward, reward, len_var, tran_matrix):
        quality = old_reward + reward
        return self.check_quality(quality)


class Quality_Identity(QualityType):
    def __init__(self, n_ops):
        super().__init__(n_ops)
    
    def calc_quality(self, old_reward, reward, len_var, tran_matrix):
        quality = reward
        return self.check_quality(quality)

class Weighted_normalised_sum(QualityType):
    """
Christian  Igel  and  Martin  Kreutz.  “Operator  adaptation  in  evolution-ary  computation  and  its  application  to  structure  optimization  of  neu-ral  networks”.  In:Neurocomputing55.1-2  (2003).https : / / ac . els -cdn.com/S0925231202006288/1-s2.0-S0925231202006288-main.pdf?_tid=c6274e78-02dc-4bf6-8d92-573ce0bed4c4&acdnat=1540907096_d0cc1e2b4ca56a49587b4d55e1008a84, pp. 347–361
"""
    def __init__(self, n_ops, decay_rate = 0.3, q_min = 0.1):
        super().__init__(n_ops)
        self.decay_rate = decay_rate
        self.q_min = q_min / self.n_ops
        #debug_print("{:>30}: decay_rate = {}, q_min = {}".format(type(self).__name__, self.decay_rate, self.q_min))
    
    def calc_quality(self, old_reward, reward, len_var, tran_matrix):
        reward += self.eps
        reward /= np.sum(reward)
        quality = self.decay_rate * np.maximum(self.q_min, reward) + (1.0 - self.decay_rate) * self.old_quality
        #if np.sum(reward) > 0:
            #reward /= np.sum(reward)
        #else:
            #reward[:] = 1.0 / self.n_ops
        #quality = self.decay_rate * reward  + (1.0 - self.decay_rate) * self.old_quality
        return super().check_quality(quality)

class Bellman_Equation(QualityType):
    """
Mudita Sharma,  Manuel Lopez-Ibanez, and  Dimitar  Kazakov. “Performance Assessment of Recursive Probability Matching for Adaptive Oper-ator Selection in Differential Evolution”. In:International Conference onParallel Problem Solving from Nature.http://eprints.whiterose.ac.uk/135483/1/paper_66_1_.pdf. Springer. 2018, pp. 321–333.
 """
    def __init__(self, n_ops, weight_reward = 1, weight_old_reward = 0.9, discount_rate = 0.01):
        super().__init__(n_ops)
        self.weight_reward = weight_reward
        self.weight_old_reward = weight_old_reward
        self.discount_rate = discount_rate
        #debug_print("{:>30}: weight_reward = {}, weight_old_reward = {}, discount_rate = {}".format(type(self).__name__, self.weight_reward, self.weight_old_reward, self.discount_rate))
    
    def calc_quality(self, old_reward, reward, len_var, tran_matrix):
        # This was called P in the original RecPM paper.
        #tran_matrix = transitive_matrix(old_probability)
        quality = self.weight_reward * reward + self.weight_old_reward * old_reward
        # Rec_PM formula:  Q_t+1 = (1 - gamma * P)^-1 x Q_t+1
        quality = np.matmul(np.linalg.pinv(1.0 - self.discount_rate * tran_matrix), quality)
        quality = softmax(quality)
        return super().check_quality(quality)



#################################################Probability definitions######################################################################

def build_probability(choice, n_ops, prob_args):
    # Use a dictionary so that we don't have to retype the name.
    choices = get_choices_dict(ProbabilityType)

    if choice == "Probability_Matching":
        return choices[choice](n_ops, prob_args["p_min"], prob_args["error_prob"])
    elif choice == "Adaptive_Pursuit":
        return choices[choice](n_ops, prob_args["p_min"], prob_args["p_max"], prob_args["learning_rate"])
    elif choice == "Probability_Identity":
        return choices[choice](n_ops)
    else:
        raise ValueError(f"probability choice {choice} unknown")
 
class ProbabilityType(BasicType):
    # Static variables
    # FIXME: Use __slots__ to find which parameters need to be defined.
    # FIXME: define this in the class as @property getter doctstring and get it from it
    params = [
        "p_min",         float,     0,    [0.0, 0.5], "Minimum probability of selection of an operator (p_min < 1 / n_ops)",
        "learning_rate", float,     0.1,    [0.0, 1.0], "Learning Rate",
        # FIXME: This should be eps_p
        "error_prob",    float,     0.0,    [0.0, 1.0], "Probability epsilon",
        "p_max",         float,     0.9,    [0.0, 1.0], "Maximum probability of selection of an operator"
    ]
    params_conditions = {
        "p_min": [],
        "learning_rate": ["Adaptive_Pursuit"],
        "error_prob": ["Probability_Matching"],
        "p_max": ["Adaptive_Pursuit"]
    }
    param_choice = "prob_choice"
    param_choice_help = "Probability method selected"
        
    def __init__(self, n_ops, p_min = None, learning_rate = None):
        # n_ops, p_min_prob and learning_rate used in more than one probability definition
        assert p_min >= 0 and p_min <= 1. / n_ops
        # FIXME: We should probably make this p_min / n_ops
        self.p_min = p_min
        self.learning_rate = learning_rate
        self.old_probability = np.full(n_ops, 1.0 / n_ops)
        # eps: a small epsilon number to avoid division by 0.
        self.eps = np.finfo(self.old_probability.dtype).eps

    def check_probability(self, probability):
        # FIXME: We already added eps in calc_probability
        probability += self.eps
        probability /= probability.sum()
        assert np.allclose(probability.sum(), 1.0, equal_nan = True)
        assert np.all(probability >= 0.0)
        # Just copy the values.
        self.old_probability[:] = probability[:]
        #debug_print("{:>30}: probability={}".format(type(self).__name__, probability))
        return probability

    @abstractmethod
    def calc_probability(self, quality):
        "Must be implemented by derived probability methods"
        pass
    
# MANUEL: These should have more descriptive names and a doctstring documenting
# where they come from (references) and what they do.
class Probability_Matching(ProbabilityType):
    # FIXME: Probability matching predates this paper.
    """Dirk Thierens. "An adaptive pursuit strategy for allocating operator probabilities".  In: Proceedings of the 7th annual conference on Genetic and evolutionary computation. http://www.cs.bham.ac.uk/~wbl/biblio/gecco2005/docs/p1539.pdf. ACM. 2005, pp. 1539–1546."""
    
    def __init__(self, n_ops, p_min = 0.1, error_prob = 0.0):
        super().__init__(n_ops, p_min = p_min)
        self.one_minus_p_min = (1.0 - n_ops * self.p_min)
        assert self.one_minus_p_min > 0.0
        self.error_prob = error_prob + self.eps
        #debug_print("{:>30}: p_min = {}, error_prob = {}".format(type(self).__name__, self.p_min, self.error_prob))
        
    def calc_probability(self, quality):
        # FIXME: Do we need epsilon above and below?
        quality += self.error_prob
        probability = normalize_sum(quality)
        if self.p_min > 0: 
            probability = self.p_min + self.one_minus_p_min * probability
        return self.check_probability(probability)
        

class Adaptive_Pursuit(ProbabilityType):
    """ Proposed by:

Dirk Thierens. “An adaptive pursuit strategy for allocating operator prob-
abilities”. In: Proceedings of the 7th annual conference on Genetic and
evolutionary computation. http://www.cs.bham.ac.uk/~wbl/biblio/gecco2005/docs/p1539.pdf. ACM. 2005, pp. 1539–1546.

"""
    def __init__(self, n_ops, p_min = 0.1, p_max = 0.9, learning_rate = 0.1):
        super().__init__(n_ops, p_min = p_min, learning_rate = learning_rate)
        self.p_max = p_max
        #debug_print("{:>30}: p_min = {}, p_max = {}, learning_rate = {}".format(type(self).__name__, self.p_min, self.p_max, self.learning_rate))

    def calc_probability(self, quality):
        delta = np.full(quality.shape[0], self.p_min)
        delta[np.argmax(quality)] = self.p_max
        probability = self.learning_rate * delta + (1.0  - self.learning_rate) * self.old_probability
        return super().check_probability(probability)

#class Adaptation_rule(ProbabilityType):
#"""
#Christian Igel and Martin Kreutz. “Using fitness distributions to improvethe evolution of learning structures”. In:Evolutionary Computation, 1999.CEC 99. Proceedings of the 1999 Congress on. Vol. 3.http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.43.2107&rep=rep1&type=pdf. IEEE. 1999, pp. 1902–1909
#"""
    #def __init__(self, n_ops, p_min = 0.025, learning_rate = 0.5):
        #super().__init__(n_ops, p_min = p_min, learning_rate = learning_rate)
        #debug_print("{:>30}: p_min = {}, learning_rate = {}".format(type(self).__name__, self.p_min, self.learning_rate))
        
    #def calc_probability(self, quality):
        # Normalize
        #quality += self.eps
        #quality /= np.sum(quality)

        # np.maximum is element-wise
        #probability = self.learning_rate * np.maximum(self.p_min, quality) + (1.0 - self.learning_rate) * self.old_probability
        #return super().check_probability(probability)

class Probability_Identity(ProbabilityType):
    def __init__(self, n_ops):
        super().__init__(n_ops)
    
    def calc_probability(self, quality):
        probability = quality
        return super().check_probability(probability)

#############Selection definitions##############################################

def build_selection(choice, n_ops, select_args, budget):
    # Use a dictionary so that we don't have to retype the name.
    choices = get_choices_dict(SelectionType)

    if choice == "Proportional_Selection":
        return choices[choice](n_ops)
    elif choice == "Greedy_Selection":
        return choices[choice](n_ops)
    elif choice == "Epsilon_Greedy_Selection":
        return choices[choice](n_ops, select_args["sel_eps"])
    elif choice == "Proportional_Greedy_Selection":
        return choices[choice](n_ops, select_args["sel_eps"])
    elif choice == "Linear_Annealed_Selection":
        return choices[choice](n_ops, budget, select_args["popsize"])
    else:
        raise ValueError(f"selection choice {choice} unknown")

class SelectionType(BasicType):
    params = ["sel_eps", float,     0.1,    [0.0, 1.0], "Random selection with probability sel_eps" ]
    params_conditions = {"sel_eps": ["Epsilon_Greedy_Selection", "Proportional_Greedy_Selection"]}

    param_choice = "select_choice"
    param_choice_help = "Selection method"
        
    def __init__(self, n_ops):
        # The initial list of operators (randomly permuted)
        self.n_ops = n_ops
        self.op_init_list = list(np.random.permutation(n_ops))

    def check_selection(self, selected):
        assert selected >= 0 and selected <= self.n_ops
        return selected
    
    @abstractmethod
    def perform_selection(self, probability, select_counter):
        pass

# MANUEL: These should have more descriptive names and a doctstring documenting
# where they come from (references) and what they do.
class Proportional_Selection(SelectionType):
    """Also called Roulette wheel selection.

 Thierens, Dirk. "An adaptive pursuit strategy for allocating operator probabilities." Proceedings of the 7th annual conference on Genetic and evolutionary computation. ACM, 2005. """
    def __init__(self, n_ops):
        super().__init__(n_ops)
    
    def perform_selection(self, probability, select_counter):
        # Roulette wheel selection
        if self.op_init_list:
            SI = self.op_init_list.pop()
        else:
            SI = np.random.choice(len(probability), p = probability)
        return super().check_selection(SI)


class Greedy_Selection(SelectionType):
    """ Fialho, Álvaro, Marc Schoenauer, and Michèle Sebag. "Toward comparison-based adaptive operator selection." Proceedings of the 12th annual conference on Genetic and evolutionary computation. ACM, 2010.
"""
    def __init__(self, n_ops):
        super().__init__(n_ops)
    
    def perform_selection(self, probability, select_counter):
        # Greedy Selection
        if self.op_init_list:
            SI = self.op_init_list.pop()
        else:
            SI = np.argmax(probability)
        return super().check_selection(SI)


class Epsilon_Greedy_Selection(SelectionType):
    # MUDITA_check: You have not checked the working of this definition, can you please check it?
    # Epsilon Greedy Selection
    def __init__(self, n_ops, sel_eps = 0.1):
        super().__init__(n_ops)
        self.sel_eps = sel_eps
        #debug_print("{:>30}: sel_eps = {}".format(type(self).__name__, self.sel_eps))
    
    def perform_selection(self, probability, select_counter):
        if self.op_init_list:
            SI = self.op_init_list.pop()
        elif np.random.uniform() < self.sel_eps:
            SI = np.random.randint(0, self.n_ops)
        else:
            SI = np.argmax(probability)
        return super().check_selection(SI)


class Proportional_Greedy_Selection(SelectionType):
    # MUDITA_check: You have not checked the working of this definition, can you please check it?
    # Combination of Proportional and Greedy Selection
    '''TODO'''
    def __init__(self, n_ops, sel_eps = 0.1):
        super().__init__(n_ops)
        self.sel_eps = sel_eps
        #debug_print("{:>30}: sel_eps = {}".format(type(self).__name__, self.sel_eps))

    def perform_selection(self, probability, select_counter):
        if self.op_init_list:
            SI = self.op_init_list.pop()
        elif np.random.uniform() < self.sel_eps:
            SI = np.random.choice(len(probability), p = probability)
        else:
            SI = np.argmax(probability)
        return super().check_selection(SI)


class Linear_Annealed_Selection(SelectionType):
    # MUDITA_check: You have not checked the working of this definition, can you please check it?
    # Linear Annealed Selection
    '''TODO'''
    def __init__(self, n_ops, budget, popsize):
        super().__init__(n_ops)
        self.budget = budget
        self.popsize = popsize
        self.n_steps = self.budget / self.popsize
        self.max_value = 1.0
        self.min_value = 0.0
        self.step_size = (self.max_value - self.min_value) / self.n_steps

    def perform_selection(self, probability, select_counter):
        self.eps_value = self.max_value - (self.step_size * select_counter)
        if self.op_init_list:
            SI = self.op_init_list.pop()
        elif np.random.uniform() < self.eps_value:
            SI = np.random.randint(0, self.n_ops)
        else:
            SI = np.argmax(probability)
        return super().check_selection(SI)

# FIXME: Find a way to call these checks at class creation time.
# Sanity checks.
RewardType.check_params()
QualityType.check_params()
ProbabilityType.check_params()
SelectionType.check_params()
AOS.check_known_AOS()

