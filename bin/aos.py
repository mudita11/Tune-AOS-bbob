from __future__ import print_function
import numpy as np
from scipy.stats import rankdata
import math
from scipy.spatial import distance
import sys
import copy

from abc import ABC,abstractmethod

# MANUEL: where is Rec_PM and the other AOS methods you implemented for the PPSN paper?
# MUDITA: Rec-PM and other are particular combination of these compenents with their default settings. They have independent folders each for an AOS (total 8 AOS).
# MANUEL: All methods should be here. If they are build with a particular combination, then create a class that instantiates that particular combination.

def debug_print(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def softmax(x):
    """TODO"""
    return np.exp(x - np.max(x))

def get_choices(cls):
    """Get all possible choices of a component of AOS framework"""
    subclasses = [x.__name__ for x in cls.__subclasses__()]
    choices = range(0, len(subclasses))
    choices_help = ', '.join("{0}:{1}".format(i,j) for i,j in zip(choices, subclasses))
    return choices, choices_help
  
def parser_add_arguments(cls, parser):
    "Helper function to add arguments of a class to an ArgumentParser"
    choices, choices_help = get_choices(cls)
    parser.add_argument("--"  + cls.arg_choice, type=int, choices=choices,
                        help=cls.arg_choice_help + " (" + choices_help + ")")
    group = parser.add_argument_group(title=cls.__name__)
    for i in range(0, len(cls.args), 3):
        arg, type, help = cls.args[i:i+3]
        group.add_argument('--' + arg, type=type, default=0, help=help)
    # Return the names
    return cls.args[0::3]

def aos_irace_parameters(cls):
    """All AOS components may call this function"""
    choices, choices_help = get_choices(cls)
    output = "# {}\n".format(cls.__name__)
    output += irace_parameter(name=cls.arg_choice, type=object, domain=choices, help=choices_help)
    for i in range(0, len(cls.args), 3):
        arg, type, help = cls.args[i:i+3]
        condition = irace_condition(cls.arg_choice, cls.args_conditions[arg])
        output += irace_parameter(name=arg, type=type, domain=cls.args_ranges[arg],
                                  condition=condition, help=help)
    return output


def irace_parameter(name, type, domain, condition="", help=""):
    """Return a string representation of an irace parameter"""
    irace_types = {int:"i", float:"r", object: "c"}
    arg = '"--{} "'.format(name)
    domain = "(" + ", ".join([str(x) for x in domain]) + ")"
    if condition != "":
        condition = "| " + condition
    if help != "":
        help = "# " + help
    return '{name:20} {arg:25} {type} {domain:20} {condition:30} {help}\n'.\
        format(name=name, arg=arg, type=irace_types[type], domain=domain,
               condition=condition, help=help)

def irace_condition(what, values):
    """Return a string representation of the condition of an irace parameter"""
    if not values:
        return ""
    if len(values) == 1:
        return what + " == " + str(values[0])
    return what + " %in% c(" + ", ".join([str(x) for x in values]) + ")"


class GenWindow(object):
    """FIXME (needs updating): gen_window stores the offspring metric data for each offspring when offspring is better than parent. It stores -1 otherwise for that offspring. Its a list. Its structre is as follows: [[[second_dim], [second_dim], [second_dim]], [[],[],[]], ...]. Second_dim has data for each offspring. Three second dim represents population size as 3, contained in third_dim. Third_dim represents a generation. Thus, this [[],[],[]] has data of all offsprings in a generation."""
    
    def __init__(self, n_ops, metric, max_gen = None):
        # Private
        self.n_ops = n_ops
        self.metric = metric - 1
        self.max_gen = max_gen
        self._gen_window_op = None
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
        for op in range(self.n_ops):
            # Assign 0.0 to any entry that is nan or belongs to a different op
            window_met = np.where((window_op == op) & ~np.isnan(window_met), window_met, 0.0)
            value[op] = function(window_met)
        return value

    def sum_at_generation(self,gen):
        """Get best metric value for all operators at generation gen"""
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

    def count_total_succ_unsucc(self, gen):
        """Counts the number of successful and unsuccessful applications for each operator in generation 'gen'"""
        window_met = self._gen_window_met[gen, :, self.metric]
        window_op = self._gen_window_op[gen, :]
        total_success = np.zeros(self.n_ops)
        total_unsuccess = np.zeros(self.n_ops)
        for op in range(self.n_ops):
            total_success[op] = np.sum((window_op == op) & np.isnan(window_met))
            total_unsuccess[op] = np.sum((window_op == op) & ~np.isnan(window_met))
        return total_success, total_unsuccess

    def metric_for_fix_appl_of_op(self, op, fix_appl):
        """Return a vector of metric values for last fix_appl applications of operator op"""
        # Stop at fix_appl starting from the end of the window (latest fix_applications of operators)
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
        off = np.tile(np.arange(ops.shape[1]).reshape(-1,1), (1, ops.shape[0]))
        out = np.hstack((ops.reshape(-1,1),
                         gen.reshape(-1,1),
                         off.reshape(-1,1),
                         met.reshape(met.shape[0] * met.shape[1], met.shape[2])))
        np.savetxt(filename, out, fmt= 3*["%d"] + 7*["%+25.18e"],
                   header = "operator generation offspring"
                   + " " + "offsp_fitness"
                   + " " + "exp_offsp_fitness"
                   + " " + "improv_wrt_parent"
                   + " " + "improv_wrt_pop"
                   + " " + "improv_wrt_bsf"
                   + " " + "improv_wrt_median"
                   + " " + "relative_fitness_improv")
                   
                   

    
class OpWindow(object):

    metrics = {"offsp_fitness": 0,
               "exp_offsp_fitness": 1,
               "improv_wrt_parent": 2,
               "improv_wrt_pop": 3,
               "improv_wrt_bsf": 4,
               "improv_wrt_median": 5,
               "relative_fitness_improv": 6
    }
    
    def __init__(self, n_ops, metric, max_size = 50):
        self.max_size = max_size
        self.n_ops = n_ops
        # FIXME: Use strings instead of numbers:
        # self.metric = self.metrics[metric]
        self.metric = metric - 1
        # Vector of operators
        self._window_op = np.full(max_size, -1)
        # Matrix of metrics
        # np.inf means not initialized
        # np.nan means unsuccessful application
        self._window_met = np.full((max_size, len(self.metrics)), np.inf)
                
    def count_ops(self):
        N = np.zeros(self.n_ops)
        op, count = np.unique(self._window_op, return_counts=True)
        N[op] = count
        return N
    
    def truncate(self, size):
        where = self.where_truncate(size)
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
        # Fill from the top
        which = (self._window_op == -1)
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


class Unknown_AOS(object):

    arg_choice = "OM_choice"
    arg_choice_help = "Offspring metric selected"

    def __init__(self, popsize, n_ops, OM_choice, rew_choice, rew_args,
                 qual_choice, qual_args, prob_choice, prob_args, select_choice):
        
        self.window = OpWindow(n_ops, metric = OM_choice, max_size = 50)
        self.gen_window = GenWindow(n_ops, metric = OM_choice)
        
        self.probability = np.full(n_ops, 1.0 / n_ops)
        rew_args["popsize"] = popsize
        self.reward_type = build_reward(rew_choice, n_ops, rew_args, self.gen_window, self.window)
        self.quality_type = build_quality(qual_choice, n_ops, qual_args, self.window)
        self.probability_type = build_probability(prob_choice, n_ops, prob_args)
        self.selection_type = build_selection(select_choice, n_ops)

    @classmethod
    def add_argument(cls, parser):
        # FIXME: Document what 1...7 means
        parser.add_argument("--" + cls.arg_choice, type=int, choices=range(1,8), help=cls.arg_choice_help)

    @classmethod
    def irace_parameters(cls):
        output = "# " + cls.__name__ + "\n"
        return output + irace_parameter(cls.arg_choice, object, range(1,8), help=cls.arg_choice_help)

    def select_operator(self):
        return self.selection_type.perform_selection(self.probability)

############################Offspring Metric definitions#######################
    def OM_Update(self, F, F1, F_bsf, opu):
        """F: fitness of parent population
        F1: fitness of children population
        F_bsf : best so far fitness
        opu: represents (op)erator that produced offspring (u).

        If offspring improves over parent, Offsping Metric (OM) stores
        (1)fitness of offspring (2)fitness improvemnt from parent to offsping
        (3)fitness improvemnt from best parent to offsping (4)fitness
        improvemnt from best so far to offsping (5)fitness improvemnt from
        median parent fitness to offsping (6)relative fitness improvemnt
        """
        F_best = np.min(F)
        F_median = np.median(F)
        eps = np.finfo(np.float32).eps
        verylarge = 10e20
        
        # See OpWindow metrics
        # Fitness is minimised but metric is maximised
        offsp_fitness = verylarge - F1
        assert np.all(offsp_fitness >= 0)
        exp_offsp_fitness = np.exp(-F1)
        improv_wrt_parent = np.fabs(F - F1)
        improv_wrt_pop = F_best - F1
        improv_wrt_bsf = F_bsf - F1
        improv_wrt_median = F_median - F1
        relative_fitness_improv = (F_bsf / (F1 + eps)) * improv_wrt_parent
        
        popsize = len(F)

        window_op = np.full(popsize, -1)
        window_met = np.full((popsize, 7), np.nan)
        
        for i in range(popsize):
            # if child is worse than parent
            if F1[i] > F[i]:
                continue
            
            window_op[i] = opu[i]
            window_met[i, 0] = offsp_fitness[i]
            window_met[i, 1] = exp_offsp_fitness[i]
            window_met[i, 2] = improv_wrt_parent[i]
            if improv_wrt_pop[i] >= 0:
                window_met[i, 3] = improv_wrt_pop[i]
            if improv_wrt_bsf[i] >= 0:
                window_met[i, 4] = improv_wrt_bsf[i]
            if improv_wrt_median[i] >= 0:
                window_met[i, 5] = improv_wrt_median[i]
                
            window_met[i, 6] = relative_fitness_improv[i]
            
            self.window.append(window_op[i], window_met[i, :])
        
        self.gen_window.append(window_op, window_met)

        reward = self.reward_type.calc_reward()
        old_reward = self.reward_type.old_reward
        old_prob = self.probability_type.old_probability
        quality = self.quality_type.calc_quality(old_reward, reward, old_prob)
        self.probability = self.probability_type.calc_probability(quality)

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
    ucb = reward + C * np.sqrt( 2 * np.log(np.sum(N)) / N)
    ucb[np.isinf(ucb) | np.isnan(ucb)] = 0
    return ucb

##################################################Reward definitions######################################################################
    


def build_reward(choice, n_ops, rew_args, gen_window, window):
    if choice == 0:
        return Pareto_Dominance(n_ops, gen_window, rew_args["fix_appl"])
    elif choice == 1:
        return Pareto_Rank(n_ops, gen_window, rew_args["fix_appl"])
    elif choice == 2:
        return Compass_projection(n_ops, gen_window, rew_args["fix_appl"], rew_args["theta"])
    elif choice == 3:
        return Area_Under_The_Curve(n_ops, window, rew_args["window_size"], rew_args["decay"])
    elif choice == 4:
        return Sum_of_Rank(n_ops, window, rew_args["window_size"], rew_args["decay"])
    elif choice == 5:
        return Success_Rate(n_ops, gen_window, rew_args["max_gen"], rew_args["succ_lin_quad"], rew_args["frac"], rew_args["noise"])
    elif choice == 6:
        return Immediate_Success(n_ops, gen_window, rew_args["popsize"])
    elif choice == 7:
        return Success_sum(n_ops, gen_window, rew_args["max_gen"])
    elif choice == 8:
        return Normalised_success_sum_window(n_ops, window, rew_args["window_size"], rew_args["normal_factor"])
    elif choice == 9:
        return Normalised_success_sum_gen(n_ops, gen_window, rew_args["max_gen"])
    elif choice == 10:
        return Best2gen(n_ops, gen_window, rew_args["scaling_constant"], rew_args["alpha"], rew_args["alpha"])
    elif choice == 11:
        return Normalised_best_sum(n_ops, gen_window, rew_args["max_gen"], rew_args["intensity"], rew_args["alpha"])
    else:
        raise ValueError("choice {} unknown".format(choice))

class RewardType(ABC):
    # Static variables
    # FIXME: Use __slots__ to find which parameters need to be defined.
    # FIXME: define this in the class as @property getter doctstring and get it from it
    args = [
        "max_gen",          int,   "Maximum number of previous generation",
        "fix_appl",         int,   "Count a fixed number of successful applications of an operator",
        "theta",            int,   "Defines search direction",
        "window_size",      int,   "Size of window",
        "decay",            float, "Decay value to emphasise the choice better operator",
        "succ_lin_quad",    int,   "Operator success as linear or quadratic",
        "frac",             float, "Fraction of sum of successes of all operators",
        "noise",            float, "Small noise for randomness",
        "normal_factor",    int,   "Choice to normalise",
        "scaling_constant", float, "Scaling constant",
        "alpha",            int,   "Choice to normalise by best produced by any operator",
        "beta",             int,   "Choice to include the difference between budget used by an operator in previous two generations",
        "intensity",        float, "Intensify the changes of best fitness value"
    ]
    args_ranges = {"max_gen": [1, 15, 30, 50],
                   "fix_appl": [50, 100, 150],
                   "theta": [36, 45, 54, 90],
                   "window_size": [20, 50],
		   "decay": [0.0, 1.0],
		   "succ_lin_quad" : [1, 2],
                   "frac": [0.0, 1.0],
                   "noise": [0.0, 1.0],
                   "normal_factor": [0, 1],
		   "scaling_constant": [0.0, 1.0],
		   "alpha" : [0, 1],
                   "beta": [0, 1],
                   "intensity": [0.0, 1.0]}

    args_conditions = {"max_gen": [5, 7, 9, 11],
                       "fix_appl": [0, 1, 2],
                       "theta": [2],
                       "window_size": [3, 4, 8],
		       "decay": [3, 4],
		       "succ_lin_quad" : [5],
                       "frac": [5],
                       "noise": [5],
                       "normal_factor": [8],
		       "scaling_constant": [10],
		       "alpha" : [10, 11],
                       "beta": [10],
                       "intensity": [11]}
    
    arg_choice = "rew_choice"
    arg_choice_help = "Reward method selected"
    
    def __init__(self, n_ops, gen_window = None, max_gen = None, window_size = None, decay = None, fix_appl = None):
        self.n_ops = n_ops
        self.gen_window = gen_window
        if max_gen:
            self.gen_window.max_gen = max_gen
        self.window_size = window_size
        self.decay = decay
        self.fix_appl = fix_appl
        self.old_reward = np.zeros(self.n_ops)
    
    @classmethod
    def add_argument(cls, parser):
        "Add arguments to an ArgumentParser"
        return parser_add_arguments(cls, parser)

    @classmethod
    def irace_parameters(cls):
        return aos_irace_parameters(cls)

    def check_reward(self, reward):
        # Nothing to check
        self.old_reward[:] = reward[:]
        debug_print("{:>30}:      reward={}".format(type(self).__name__, reward))
        return reward

    @abstractmethod
    def calc_reward(self):
        pass

class Pareto_Dominance(RewardType):
    """
Jorge Maturana, Fr ́ed ́eric Lardeux, and Frederic Saubion. “Autonomousoperator management for evolutionary algorithms”. In:Journal of Heuris-tics16.6 (2010).https://link.springer.com/content/pdf/10.1007/s10732-010-9125-3.pdf, pp. 881–909.
"""

    def __init__(self, n_ops, gen_window, fix_appl = 20):
        super().__init__(n_ops, gen_window = gen_window, fix_appl = fix_appl)
        debug_print("{:>30}: fix_appl = {}".format(type(self).__name__, self.fix_appl))
    
    def calc_reward(self):
        # Pareto dominance returns the number of operators dominated by an
        # operator whereas Pareto rank gives the number of operators an
        # operator is dominated by.
        std_op = np.full(self.n_ops, np.nan)
        mean_op = np.full(self.n_ops, np.nan)
        for i in range(self.n_ops):
            b = self.gen_window.metric_for_fix_appl_of_op(i, self.fix_appl)
            if len(b) > 0:
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
        if np.sum(reward) != 0:
            reward /= np.sum(reward)
        return super().check_reward(reward)


class Pareto_Rank(RewardType):
    """
Jorge Maturana, Fr ́ed ́eric Lardeux, and Frederic Saubion. “Autonomous operator management for evolutionary algorithms”. In:Journal of Heuris-tics16.6 (2010).https://link.springer.com/content/pdf/10.1007/s10732-010-9125-3.pdf, pp. 881–909.
"""
    def __init__(self, n_ops, gen_window, fix_appl = 20):
        super().__init__(n_ops, gen_window = gen_window, fix_appl = fix_appl)
        debug_print("{:>30}: fix_appl = {}".format(type(self).__name__, self.fix_appl))

    def calc_reward(self):
        std_op = np.full(self.n_ops, np.nan)
        mean_op = np.full(self.n_ops, np.nan)
        for i in range(self.n_ops):
            b = self.gen_window.metric_for_fix_appl_of_op(i, self.fix_appl)
            if len(b) > 0:
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
        
        if np.sum(reward) != 0:
            reward /= np.sum(reward)
        reward = 1. - reward
        return super().check_reward(reward)


class Compass_projection(RewardType):
    """
        Jorge Maturana and Fr ́ed ́eric Saubion. “A compass to guide genetic al-gorithms”. In:International Conference on Parallel Problem Solving fromNature.http://www.info.univ-angers.fr/pub/maturana/files/MaturanaSaubion-Compass-PPSNX.pdf. Springer. 2008, pp. 256–265.
        """
    def __init__(self, n_ops, gen_window, fix_appl = 100, theta = 45):
        super().__init__(n_ops, gen_window = gen_window, fix_appl = fix_appl)
        self.theta = theta
        debug_print("{:>30}: fix_appl = {}".format(type(self).__name__, self.fix_appl, self.theta))
    
    def calc_reward(self):
        reward = np.zeros(self.n_ops)
        # Projection on line B with thetha = pi/4
        #        B = [1, 1]
        for i in range(self.n_ops):
            b = self.gen_window.metric_for_fix_appl_of_op(i, self.fix_appl)
            if len(b) > 0:
                # Diversity
                std = np.std(b)
                # Quality 
                avg = np.mean(b)
                # abs(atan(Quality / Diversity) - theta)
                angle = np.fabs(np.arctan(np.deg2rad(avg / std)) - self.theta)
                # Euclidean distance of the vector 
                reward[i] = (np.sqrt(std**2 + avg**2)) * np.cos(angle)
                # Maturana & Sablon (2008) divide by T_it defined as mean
                # execution time of operator i over its last t applications.
                # We do not divide
        reward = reward - np.min(reward)
        return super().check_reward(reward)

class Area_Under_The_Curve(RewardType):
    """
Alvaro Fialho, Marc Schoenauer, and Mich`ele Sebag. “Toward comparison-based adaptive operator selection”. In:Proceedings of the 12th annual con-ference on Genetic and evolutionary computation.https://hal.inria.fr/file/index/docid/471264/filename/banditGECCO10.pdf. ACM.2010, pp. 767–774
"""
    def __init__(self, n_ops, window, window_size = 50, decay = 0.4):
        super().__init__(n_ops, window_size = window_size, decay = decay)
        self.window = window
        self.window_size = window_size
        debug_print("{:>30}: window_size = {}, decay = {}".format(type(self).__name__, self.window_size, self.decay))
    
    def calc_reward(self):
        reward = np.zeros(self.n_ops)
        #print("Window in AUC calc ", self.window, type(self.window), np.shape(self.window))
        window = self.window.truncate(self.window_size)
        window_op_sorted, rank = window.get_ops_sorted_and_rank()
        for op in range(self.n_ops):
            reward[op] = AUC(window_op_sorted, rank, op, self.window_size, self.decay)
            # print("Inside reward: ", reward)
        return super().check_reward(reward)

class Sum_of_Rank(RewardType):
    """
Alvaro Fialho, Marc Schoenauer, and Mich`ele Sebag. “Toward comparison-based adaptive operator selection”. In:Proceedings of the 12th annual con-ference on Genetic and evolutionary computation.https://hal.inria.fr/file/index/docid/471264/filename/banditGECCO10.pdf. ACM.2010, pp. 767–774
"""
    def __init__(self, n_ops, window, window_size = 50, decay = 0.4):
        super().__init__(n_ops, window_size = window_size, decay = decay)
        self.window = window
        self.window_size = window_size
        debug_print("{:>30}: window_size = {}, decay = {}".format(type(self).__name__, self.window_size, self.decay))
    
    def calc_reward(self):
        reward = np.zeros(self.n_ops)
        window = self.window.truncate(self.window_size)
        window_op_sorted, rank = window.get_ops_sorted_and_rank()
        # Fialho's thesis: https://tel.archives-ouvertes.fr/tel-00578431/document (pg. 79).
        value = (self.decay ** rank) * (self.window_size - rank)
        for i in range(self.n_ops):
            reward[i] = value[window_op_sorted == i].sum()
        if np.sum(reward) != 0:
            reward /= np.sum(reward)
        return super().check_reward(reward)



class Success_Rate(RewardType):
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
    
    def __init__(self, n_ops, gen_window, max_gen = 10, succ_lin_quad = 1, frac = 0.01, noise = 0.0):
        super().__init__(n_ops, gen_window = gen_window, max_gen = max_gen)
        self.succ_lin_quad = succ_lin_quad
        self.frac = frac
        self.noise = noise
        debug_print("{:>30}: max_gen = {}, succ_lin_quad = {}, frac = {}, noise = {}".format(type(self).__name__, self.gen_window.max_gen, self.succ_lin_quad, self.frac, self.noise))
    
    def calc_reward(self):
        gen_window_len = len(self.gen_window)
        max_gen = self.gen_window.get_max_gen()
        reward = np.zeros(self.n_ops)
        for j in range(gen_window_len - max_gen, gen_window_len):
            total_success, total_unsuccess = self.gen_window.count_total_succ_unsucc(j)
            napplications = total_success + total_unsuccess
            # Avoid division by zero. If total == 0, then total_success is zero.
            napplications[napplications == 0] = 1
            reward += (total_success ** self.succ_lin_quad + self.frac * np.sum(total_success)) / napplications
        reward += self.noise
        return super().check_reward(reward)


class Immediate_Success(RewardType):
    """
 Mudita  Sharma,  Manuel  L ́opez-Ib ́a ̃nez,  and  Dimitar  Kazakov.  “Perfor-mance Assessment of Recursive Probability Matching for Adaptive Oper-ator Selection in Differential Evolution”. In:International Conference onParallel Problem Solving from Nature.http://eprints.whiterose.ac.uk/135483/1/paper_66_1_.pdf. Springer. 2018, pp. 321–333.
y """
    def __init__(self, n_ops, gen_window, popsize):
        super().__init__(n_ops, gen_window = gen_window)
        self.popsize = popsize
    
    def calc_reward(self):
        gen_window_len = len(self.gen_window)
        total_success, total_unsuccess = self.gen_window.count_total_succ_unsucc(gen_window_len - 1)
        reward = total_success / self.popsize
        return super().check_reward(reward)

class Success_sum(RewardType):
    """
 Christian  Igel  and  Martin  Kreutz.  “Operator  adaptation  in  evolution-ary  computation  and  its  application  to  structure  optimization  of  neu-ral  networks”.  In:Neurocomputing55.1-2  (2003).https : / / ac . els -cdn.com/S0925231202006288/1-s2.0-S0925231202006288-main.pdf?_tid=c6274e78-02dc-4bf6-8d92-573ce0bed4c4&acdnat=1540907096_d0cc1e2b4ca56a49587b4d55e1008a84, pp. 347–361.
 """
    def __init__(self, n_ops, gen_window, max_gen = 4):
        super().__init__(n_ops, gen_window = gen_window, max_gen = max_gen)
        debug_print("{:>30}: max_gen = {}".format(type(self).__name__, self.gen_window.max_gen))
    
    def calc_reward(self):
        gen_window_len = len(self.gen_window)
        max_gen = self.gen_window.get_max_gen()
        napplications = np.zeros(self.n_ops)
        reward = np.zeros(self.n_ops)
        for j in range(gen_window_len - max_gen, gen_window_len):
            total_success, total_unsuccess = self.gen_window.count_total_succ_unsucc(j)
            napplications += total_success + total_unsuccess
            value = self.gen_window.sum_at_generation(j)
            reward += value
        napplications[napplications == 0] = 1
        reward /= napplications
        return super().check_reward(reward)

class Normalised_success_sum_window(RewardType):
    """
Alvaro Fialho, Marc Schoenauer, and Mich`ele Sebag. “Analysis of adaptiveoperator selection techniques on the royal road and long k-path problems”.In:Proceedings of the 11th Annual conference on Genetic and evolutionarycomputation.https://hal.archives-ouvertes.fr/docs/00/37/74/49/PDF/banditGECCO09.pdf. ACM. 2009, pp. 779–786.
"""
    def __init__(self, n_ops, window, window_size = 50, normal_factor = 0.1):
        super().__init__(n_ops, window_size = window_size)
        self.window = window
        self.window_size = window_size
        self.normal_factor = normal_factor
        debug_print("{:>30}: window_size = {}, normal_factor = {}".
                    format(type(self).__name__, self.window_size, self.normal_factor))
    
    def calc_reward(self):
        reward = np.zeros(self.n_ops)
        # Create a local truncated window.
        window = self.window.truncate(self.window_size)
        N = window.count_ops()
        N[N == 0] = 1
        reward = window.sum_per_op() / N
        if np.max(reward) != 0:
            reward /= np.max(reward)**self.normal_factor
        return super().check_reward(reward)

class Normalised_success_sum_gen(RewardType):
    """
Christian Igel and Martin Kreutz. “Using fitness distributions to improvethe evolution of learning structures”. In:Evolutionary Computation, 1999.CEC 99. Proceedings of the 1999 Congress on. Vol. 3.http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.43.2107&rep=rep1&type=pdf. IEEE. 1999, pp. 1902–1909.
"""
    def __init__(self, n_ops, gen_window, max_gen = 4):
        super().__init__(n_ops, gen_window = gen_window, max_gen = max_gen)
        debug_print("{:>30}: max_gen = {}".format(type(self).__name__, self.gen_window.max_gen))
    
    def calc_reward(self):
        gen_window_len = len(self.gen_window)
        max_gen = self.gen_window.get_max_gen()
        reward = np.zeros(self.n_ops)
        for j in range(gen_window_len - max_gen, gen_window_len):
            total_success, total_unsuccess = self.gen_window.count_total_succ_unsucc(j)
            napplications = total_success + total_unsuccess
            napplications[napplications == 0] = 1
            value = self.gen_window.sum_at_generation(j)
            reward += value / napplications
            
        return super().check_reward(reward)

class Best2gen(RewardType):
    """
Giorgos Karafotias, Agoston Endre Eiben, and Mark Hoogendoorn. “Genericparameter  control  with  reinforcement  learning”.  In:Proceedings of the2014 Annual Conference on Genetic and Evolutionary Computation.http://www.few.vu.nl/~gks290/papers/GECCO2014-RLControl.pdf. ACM.2014, pp. 1319–1326.
 """
    def __init__(self, n_ops, gen_window, scaling_constant = 1, alpha = 0, beta = 1):
        super().__init__(n_ops, gen_window)
        self.scaling_constant = scaling_constant
        self.alpha = alpha
        self.beta = beta
        debug_print("{:>30}: scaling constant = {}, alpha = {}, beta = {}".format(type(self).__name__, self.scaling_constant, self.alpha, self.beta))
    
    def calc_reward(self):
        # Involves calculation of best in previous two generations.
        gen_window_len = len(self.gen_window)
        total_success_t, total_unsuccess_t = self.gen_window.count_total_succ_unsucc(gen_window_len - 1)
        if gen_window_len >= 2:
            total_success_t_1, total_unsuccess_t_1 = self.gen_window.count_total_succ_unsucc(gen_window_len - 2)
        else:
            total_success_t_1 = 0
            total_unsuccess_t_1 = 0
        n_applications = (total_success_t + total_unsuccess_t) - (total_success_t_1 + total_unsuccess_t_1)
        n_applications[n_applications == 0] = 1
        
        # Calculating best in current generation
        best_t = self.gen_window.max_at_generation(gen_window_len - 1)
        # Calculating best in last generation
        if gen_window_len >= 2:
            best_t_1 = self.gen_window.max_at_generation(gen_window_len - 2)
            best_t_1[best_t_1 == 0] = 1
        else:
            best_t_1 = np.ones(self.n_ops)
            
        reward = self.scaling_constant * np.fabs(best_t - best_t_1) / ((best_t_1**self.alpha) * (np.fabs(n_applications)**self.beta))
        return super().check_reward(reward)

class Normalised_best_sum(RewardType):
    """
Alvaro Fialho, Marc Schoenauer, and Mich`ele Sebag. “Analysis of adaptiveoperator selection techniques on the royal road and long k-path problems”.In:Proceedings of the 11th Annual conference on Genetic and evolutionarycomputation.https://hal.archives-ouvertes.fr/docs/00/37/74/49/PDF/banditGECCO09.pdf. ACM. 2009, pp. 779–786.
"""
    def __init__(self, n_ops, gen_window, max_gen = 10, intensity = 0, alpha = 1):
        super().__init__(n_ops, gen_window = gen_window, max_gen = max_gen)
        self.intensity = intensity
        self.alpha = alpha
        debug_print("{:>30}: max_gen = {}, intensity = {}, alpha = {}".format(
            type(self).__name__, self.gen_window.max_gen, self.intensity, self.alpha))
    
    def calc_reward(self):
        # Normalised best sum
        reward = np.zeros(self.n_ops)
        max_gen = self.gen_window.get_max_gen()
        for i in range(self.n_ops):
            reward[i] = np.sum(self.gen_window.max_per_generation(i))
        reward = (1.0 / max_gen) * (reward**self.intensity) / (np.max(reward)**self.alpha)
        return super().check_reward(reward)


##################################################Quality definitions######################################################################

def build_quality(choice, n_ops, qual_args, window):
    if choice == 0:
        return Weighted_sum(n_ops, qual_args["decay_rate"])
    elif choice == 1:
        return Upper_confidence_bound(n_ops, window, qual_args["scaling_factor"])
    elif choice == 2:
        return Identity(n_ops)
    elif choice == 3:
        return Weighted_normalised_sum(n_ops, qual_args["decay_rate"])
    elif choice == 4:
        return Markov_reward_process(n_ops, qual_args["weight_reward"], qual_args["weight_old_reward"], qual_args["discount_rate"])
    else:
        raise ValueError("choice {} unknown".format(choice))


class QualityType(ABC):
    # Static variables
    # FIXME: Use __slots__ to find which parameters need to be defined.
    # FIXME: define this in the class as @property getter doctstring and get it from it
    args = [
        "scaling_factor",    float,"Scaling Factor",            
        "decay_rate",        float,"Decay rate",                
        "weight_reward",     float,"Memory for current reward", 
        "weight_old_reward", float,"Memory for previous reward", 
        "discount_rate",     float,"Discount rate"
    ]
    # TODO:
    args_ranges = {"scaling_factor" : [0.0, 1.0],
                   "decay_rate": [0.0, 1.0],
                   "weight_reward": [0.0, 1.0],
                   "weight_old_reward": [0.0, 1.0],
		   "discount_rate": [0.0, 1.0]}
    # TODO:
    args_conditions = {"scaling_factor" : [1],
                       "decay_rate": [0, 3],
                       "weight_reward": [4],
                       "weight_old_reward": [4],
		       "discount_rate": [4]}

    arg_choice = "qual_choice"
    arg_choice_help = "Quality method selected"
    
    def __init__(self, n_ops):
        self.n_ops = n_ops
        self.old_quality = np.zeros(n_ops)
        
    @classmethod
    def add_argument(cls, parser):
        "Add arguments to an ArgumentParser"
        return parser_add_arguments(cls, parser)

    @classmethod
    def irace_parameters(cls):
        return aos_irace_parameters(cls)

    def check_quality(self, quality):
        assert np.sum(quality) >= 0
        if np.sum(quality) != 0:
            quality /= np.sum(quality)
        assert np.all(quality >= 0.0)
        self.old_quality[:] = quality[:]
        debug_print("{:>30}:     quality={}".format(type(self).__name__, quality))
        return(quality)
    
    @abstractmethod
    def calc_quality(self, old_reward, reward):
        pass

# MANUEL: These should have more descriptive names and a doctstring documenting
# where they come from (references) and what they do.
class Weighted_sum(QualityType):
    """
 Dirk Thierens. “An adaptive pursuit strategy for allocating operator prob-abilities”.  In:Proceedings of the 7th annual conference on Genetic andevolutionary computation.http://www.cs.bham.ac.uk/~wbl/biblio/gecco2005/docs/p1539.pdf. ACM. 2005, pp. 1539–1546.
 """
    def __init__(self, n_ops, decay_rate = 0.6):
        super().__init__(n_ops)
        self.decay_rate = decay_rate
        debug_print("{:>30}: decay_rate = {}".format(type(self).__name__, self.decay_rate))
    
    def calc_quality(self, old_reward, reward, old_probability):
        quality = self.decay_rate * reward + (1.0 - self.decay_rate) * self.old_quality
        return super().check_quality(quality)

class Upper_confidence_bound(QualityType):
    """
Alvaro Fialho et al. “Extreme value based adaptive operator selection”.In:International Conference on Parallel Problem Solving from Nature.https : / / hal . inria . fr / file / index / docid / 287355 / filename /rewardPPSN.pdf. Springer. 2008, pp. 175–184
"""
    def __init__(self, n_ops, window, scaling_factor = 0.5):
        super().__init__(n_ops)
        self.window = window
        self.scaling_factor = scaling_factor
        debug_print("{:>30}: scaling_factor = {}".format(type(self).__name__, self.scaling_factor))
    
    def calc_quality(self, old_reward, reward, old_probability):
        #window_op_sorted, N, rank = count_op(self.n_ops, self.window, self.off_met)
        N = self.window.count_ops()
        quality = UCB(N, self.scaling_factor, reward)
        return super().check_quality(quality)

class Identity(QualityType):
    def __init__(self, n_ops):
        super().__init__(n_ops)
    
    def calc_quality(self, old_reward, reward, old_probability):
        quality = reward
        return super().check_quality(quality)

class Weighted_normalised_sum(QualityType):
    """
Christian  Igel  and  Martin  Kreutz.  “Operator  adaptation  in  evolution-ary  computation  and  its  application  to  structure  optimization  of  neu-ral  networks”.  In:Neurocomputing55.1-2  (2003).https : / / ac . els -cdn.com/S0925231202006288/1-s2.0-S0925231202006288-main.pdf?_tid=c6274e78-02dc-4bf6-8d92-573ce0bed4c4&acdnat=1540907096_d0cc1e2b4ca56a49587b4d55e1008a84, pp. 347–361
"""
    def __init__(self, n_ops, decay_rate = 0.3):
        super().__init__(n_ops)
        self.decay_rate = decay_rate
        debug_print("{:>30}: decay_rate = {}".format(type(self).__name__, self.decay_rate))
    
    def calc_quality(self, old_reward, reward, old_probability):
        if np.sum(reward) > 0:
            reward /= np.sum(reward)
        else:
            reward[:] = 1.0 / self.n_ops
        quality = self.decay_rate * reward  + (1.0 - self.decay_rate) * self.old_quality
        return super().check_quality(quality)

class Markov_reward_process(QualityType):
    """
Mudita Sharma,  Manuel Lopez-Ibanez, and  Dimitar  Kazakov. “Performance Assessment of Recursive Probability Matching for Adaptive Oper-ator Selection in Differential Evolution”. In:International Conference onParallel Problem Solving from Nature.http://eprints.whiterose.ac.uk/135483/1/paper_66_1_.pdf. Springer. 2018, pp. 321–333.
 """
    def __init__(self, n_ops, weight_reward = 1, weight_old_reward = 0.9, discount_rate = 0.0):
        super().__init__(n_ops)
        self.weight_reward = weight_reward
        self.weight_old_reward = weight_old_reward
        self.discount_rate = discount_rate
        debug_print("{:>30}: weight_reward = {}, weight_old_reward = {}, discount_rate = {}".format(type(self).__name__, self.weight_reward, self.weight_old_reward, self.discount_rate))
    
    def calc_quality(self, old_reward, reward, old_probability):
        # This was called P in the original RecPM paper.
        tran_matrix = transitive_matrix(old_probability)
        quality = self.weight_reward * reward + self.weight_old_reward * old_reward
        # Rec_PM formula:  Q_t+1 = (1 - gamma * P)^-1 x Q_t+1
        quality = np.matmul(np.linalg.pinv(1.0 - self.discount_rate * tran_matrix), quality)
        quality = softmax(quality)
        return super().check_quality(quality)



#################################################Probability definitions######################################################################

def build_probability(choice, n_ops, prob_args):
    if choice == 0:
        return Probability_Matching(n_ops, prob_args["p_min"], prob_args["error_prob"])
    elif choice == 1:
        return Adaptive_Pursuit(n_ops, prob_args["p_min"], prob_args["p_max"], prob_args["learning_rate"])
    elif choice == 2:
        return Adaptation_rule(n_ops, prob_args["p_min"], prob_args["learning_rate"])
    else:
        raise ValueError("choice {} unknown".format(choice))
 
class ProbabilityType(ABC):
    # Static variables
    # FIXME: Use __slots__ to find which parameters need to be defined.
    # FIXME: define this in the class as @property getter doctstring and get it from it
    args = [
        "p_min",         float,"Minimum probability of selection of an operator",
        "learning_rate", float,"Learning Rate",                                 
        "error_prob",    float,"Probability noise",                             
        "p_max",         float,"Maximum probability of selection of an operator"
    ]
    args_ranges = {"p_min" : [0.0, 0.25],
                   "learning_rate": [0.0, 1.0],
                   "error_prob": [0.0, 1.0],
                   "p_max": [0.0, 1.0]}
    # FIXME: We should use explicit class names or a function that converts from names to numbers
    args_conditions = {"p_min": [],
                       "learning_rate": [1,2],
                       "error_prob": [0],
                       "p_max": [1]}
        
    arg_choice = "prob_choice"
    arg_choice_help = "Probability method selected"

    def __init__(self, n_ops, p_min = None, learning_rate = None):
        # n_ops, p_min_prob and learning_rate used in more than one probability definition
        self.p_min = p_min
        assert self.p_min != 1.0 / n_ops
        self.learning_rate = learning_rate
        self.old_probability = np.full(n_ops, 1.0 / n_ops)
        self.eps = np.finfo(self.old_probability.dtype).eps

    @classmethod
    def add_argument(cls, parser):
        "Add arguments to an ArgumentParser"
        return parser_add_arguments(cls, parser)

    @classmethod
    def irace_parameters(cls):
        return aos_irace_parameters(cls)

    def check_probability(self, probability):
        probability += self.eps
        probability /= np.sum(probability)
        assert np.allclose(np.sum(probability), 1.0, equal_nan = True)
        assert np.all(probability >= 0.0)
        # Just copy the values.
        self.old_probability[:] = probability[:]
        debug_print("{:>30}: probability={}".format(type(self).__name__, probability))
        return(probability)

    @abstractmethod
    def calc_probability(self, quality):
        "Must be implemented by derived probability methods"
        pass
    
# MANUEL: These should have more descriptive names and a doctstring documenting
# where they come from (references) and what they do.
class Probability_Matching(ProbabilityType):
    """
 Dirk Thierens. “An adaptive pursuit strategy for allocating operator prob-abilities”.  In:Proceedings of the 7th annual conference on Genetic andevolutionary computation.http://www.cs.bham.ac.uk/~wbl/biblio/gecco2005/docs/p1539.pdf. ACM. 2005, pp. 1539–1546.
"""
    def __init__(self, n_ops, p_min = 0.1, error_prob = 0.0):
        super().__init__(n_ops, p_min = p_min)
        # np.finfo(np.float32).eps adds a small epsilon number that doesn't make any difference but avoids 0.
        self.error_prob = error_prob + self.eps
        debug_print("{:>30}: p_min = {}, error_prob = {}".format(type(self).__name__, self.p_min, self.error_prob))
        
    def calc_probability(self, quality):
        quality += self.error_prob
        probability = self.p_min + (1 - len(quality) * self.p_min) * (quality / np.sum(quality))
        return super().check_probability(probability)
        

class Adaptive_Pursuit(ProbabilityType):
    """ Proposed by:

Dirk Thierens. “An adaptive pursuit strategy for allocating operator prob-
abilities”. In: Proceedings of the 7th annual conference on Genetic and
evolutionary computation. http://www.cs.bham.ac.uk/~wbl/biblio/gecco2005/docs/p1539.pdf. ACM. 2005, pp. 1539–1546.

"""
    def __init__(self, n_ops, p_min = 0.1, p_max = 0.9, learning_rate = 0.1):
        super().__init__(n_ops, p_min = p_min, learning_rate = learning_rate)
        self.p_max = p_max
        debug_print("{:>30}: p_min = {}, p_max = {}, learning_rate = {}".format(type(self).__name__, self.p_min, self.p_max, self.learning_rate))

    def calc_probability(self, quality):
        delta = np.full(quality.shape[0], self.p_min)
        delta[np.argmax(quality)] = self.p_max
        probability = self.learning_rate * delta + (1.0  - self.learning_rate) * self.old_probability
        return super().check_probability(probability)

class Adaptation_rule(ProbabilityType):
    """
Christian Igel and Martin Kreutz. “Using fitness distributions to improvethe evolution of learning structures”. In:Evolutionary Computation, 1999.CEC 99. Proceedings of the 1999 Congress on. Vol. 3.http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.43.2107&rep=rep1&type=pdf. IEEE. 1999, pp. 1902–1909
"""
    def __init__(self, n_ops, p_min = 0.025, learning_rate = 0.5):
        super().__init__(n_ops, p_min = p_min, learning_rate = learning_rate)
        debug_print("{:>30}: p_min = {}, learning_rate = {}".format(type(self).__name__, self.p_min, self.learning_rate))
        
    def calc_probability(self, quality):
        # Normalize
        quality += self.eps
        quality /= np.sum(quality)

        # np.maximum is element-wise
        probability = self.learning_rate * np.maximum(self.p_min, quality) + (1.0 - self.learning_rate) * self.old_probability
        return super().check_probability(probability)


#############Selection definitions##############################################

def build_selection(choice, n_ops):
    if choice == 0:
        return Proportional_Selection(n_ops)
    elif choice == 1:
        return Greedy_Selection(n_ops)
    else:
        raise ValueError("choice {} unknown".format(choice))

class SelectionType(ABC):

    args = []
    arg_choice = "select_choice"
    arg_choice_help = "Selection method"

    def __init__(self, n_ops):
        # The initial list of operators (randomly permuted)
        self.n_ops = n_ops
        self.op_init_list = list(np.random.permutation(n_ops))

    @classmethod
    def add_argument(cls, parser):
        "Add arguments to an ArgumentParser"
        return parser_add_arguments(cls, parser)

    @classmethod
    def irace_parameters(cls):
        return aos_irace_parameters(cls)

    def check_selection(self, selected):
        assert selected >= 0 and selected <= self.n_ops
        return selected
    
    @abstractmethod
    def perform_selection(self, probability):
        pass

# MANUEL: These should have more descriptive names and a doctstring documenting
# where they come from (references) and what they do.
class Proportional_Selection(SelectionType):
    '''TODO'''
    def __init__(self, n_ops):
        super().__init__(n_ops)
    
    def perform_selection(self, probability):
        # Roulette wheel selection
        if self.op_init_list:
            SI = self.op_init_list.pop()
        else:
            SI = np.random.choice(len(probability), p = probability)
        return super().check_selection(SI)


class Greedy_Selection(SelectionType):
    '''TODO'''
    def __init__(self, n_ops):
        super().__init__(n_ops)
    
    def perform_selection(self, probability):
        # Greedy Selection
        if self.op_init_list:
            SI = self.op_init_list.pop()
        else:
            SI = np.argmax(probability)
        return super().check_selection(SI)

