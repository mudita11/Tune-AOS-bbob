from __future__ import print_function
import numpy as np
from scipy.stats import rankdata
import math
from scipy.spatial import distance
import sys

from abc import ABC,abstractmethod

# MANUEL: where is Rec_PM and the other AOS methods you implemented for the PPSN paper?
# MUDITA: Rec-PM and other are particular combination of these compenents with their default settings. They have independent folders each for an AOS (total 8 AOS).
# MANUEL: All methods should be here. If they are build with a particular combination, then create a class that instantiates that particular combination.

def debug_print(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def softmax(x):
    """TODO"""
    x = x - np.max(quality)
    x = np.exp(x)
    return(x)

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
    output += irace_parameter(name=cls.arg_choice, type=str, domain=choices, help=choices_help)
    for i in range(0, len(cls.args), 3):
        arg, type, help = cls.args[i:i+3]
        condition = irace_condition(cls.arg_choice, cls.args_conditions[arg])
        output += irace_parameter(name=arg, type=type, domain=cls.args_ranges[arg],
                                  condition=condition, help=help)
    return output


def irace_parameter(name, type, domain, condition="", help=""):
    """Return a string representation of an irace parameter"""
    irace_types = {int:"i", float:"r", str: "c"}
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

import copy

class OpWindow():

    metrics = {"a": 0, "b": 1, "c": 2, "d": 3}
    
    def __init__(self, n_ops, metric, max_size = 50):
        self.max_size = max_size
        self.n_ops = n_ops
        self.metric = self.metrics[metric]
        # Vector of operators
        self._window_op = np.full(max_size, -1)
        # Matrix of metrics
        self._window_met = np.full((max_size, len(self.metrics)), np.nan)

    def truncate(self, size):
        where = self.where_truncate(size)
        truncated = copy.copy(self)
        truncated._window_op = truncated._window_op[where]
        truncated._window_met = truncated._window_met[where, :]
        return truncated
    
    def where_truncate(self, size):
        """Returns the indexes of a truncated window after removing the offspring entry with unimproved metric from window and truncating to size"""
        assert size > 0
        where = np.where(np.isfinite(self._window_met[:, self.metric]))
        return where[:size]

    def sum_per_op(self):
        # FIXME: there is probably a faster way to do this.
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
        # MUDITA: Loop for filling the window as improved offsping are generated
        # MANUEL: What is window? You sometimes index it like a list window[][] and other times like a matrix [, ]
        if np.any(self.window[:, 1] == np.inf):
            for value in range(self.max_window_size - 1, -1, -1):
                if self.window[value][0] == -1:
                    self.window[value] = second_dim
                    #print(self.window)
                    break
        else:
            # MANUEL: What is this code doing? It is not obvious.
            # MUDITA: When a new sample comes in (second dimension), it finds from the last index a sample to replace that was generated by the same operator application. Once it gets the index of the sample to replace (nn), it removes it and shifts all samples one index below to put that new sample at top of window (index = 0). In case it could not find any existing application of that operator in the window, it replaces the worst candidate.
            for nn in range(self.max_window_size-1,-1,-1):
                if self.window[nn][0] == self.opu[i] or nn == 0:
                    for nn1 in range(nn, 0, -1):
                        self.window[nn1] = self.window[nn1-1]
                    self.window[0] = second_dim
                    break
#               elif nn==0 and self.window[nn][0] != self.opu[i]:
                    # MANUEL: What is window? You sometimes index it like a list window[][] and other times like a matrix [, ].
#                   if self.F1[i] < np.max(self.window[: ,1]):
#                       self.window[np.argmax(self.window[:,1])] = second_dim
        which = self._window_op == op
        if len(which) == 0:
            self._window_op[]

    def get_metric(self, metric = self.metric):
        assert metric >= 0 && metric < len(self.metrics)
        return(self._window_met[:, metric])

    def get_ops(self):
        return(self._window_op)

# MANUEL: What is the difference between AOS and unknown AOS?
# MUDITA: I am referring to a combination of these components as Unknown AOS if that combination is not considered in literature.
# MANUEL: I don't understand. What do you use AOS for?
# MUDITA: I think more appropriate name for AOS class is AOS_Update because this class is basically updating components of AOS. Its not an AOS method.

class Unknown_AOS(object):

    arg_choice = "OM_choice"
    arg_choice_help = "Offspring metric selected"

    def __init__(self, popsize, F1, F, u, X, f_min, x_min, best_so_far, n_ops, OM_choice, rew_choice, rew_args, qual_choice, qual_args, prob_choice, prob_args, select_choice):
        self.popsize = int(popsize)
        self.F1 = np.array(F1)
        self.F = np.array(F)
        self.u = u
        self.X = X
        self.f_min = f_min
        self.x_min = x_min
        self.best_so_far = best_so_far
        self.n_ops = n_ops
        
        # MANUEL: What is opu?
        # MUDITA: opu represents (op)erator that produced offspring (u).
        self.opu = np.full(self.popsize, 4)
        self.old_opu = self.opu.copy()
        self.window = OpWindow(n_ops, max_size = 50, metric = OM_choice)
        
        # MANUEL: What are these?
        # MUDITA: gen_window stores the offspring metric data for each offspring when offspring is better than parent. It stores -1 otherwise for that offspring. Its a list. Its structre is as follows: [[[second_dim], [second_dim], [second_dim]], [[],[],[]], ...]. Second_dim has data for each offspring. Three second dim represents population size as 3, contained in third_dim. Third_dim represents a generation. Thus, this [[],[],[]] has data of all offsprings in a generation.
        self.gen_window = [] # print("Inside AOS", type(self.gen_window), self.gen_window)
        
        self.reward = np.zeros(self.n_ops)
        self.old_reward = self.reward.copy()
        rew_args["popsize"] = popsize
        self.reward_type = build_reward(rew_choice, n_ops, rew_args, self.gen_window, self.window, OM_choice)
        
        self.quality = np.full(n_ops, 1.0)
        self.quality_type = build_quality(qual_choice, n_ops, qual_args, self.window, OM_choice)

        self.probability = np.full(n_ops, 1.0 / n_ops)
        self.probability_type = build_probability(prob_choice, n_ops, prob_args)
        self.selection_type = build_selection(select_choice, n_ops)
    

    @classmethod
    def add_argument(cls, parser):
        # FIXME: Document what 1...7 means
        parser.add_argument("--" + cls.arg_choice, type=int, choices=range(1,8), help=cls.arg_choice_help)

    @classmethod
    def irace_parameters(cls):
        output = "# " + cls.__name__ + "\n"
        return output + irace_parameter(cls.arg_choice, str, range(1,8), help=cls.arg_choice_help)

##################################################Offspring Metric definitions##################################################################
    def OM_Update(self):
        """If offspring improves over parent, Offsping Metric (OM) stores (1)fitness of offspring (2)fitness improvemnt from parent to offsping (3)fitness improvemnt from best parent to offsping (4)fitness improvemnt from best so far to offsping (5)fitness improvemnt from median parent fitness to offsping (6)relative fitness improvemnt """
        third_dim = []
        F_min = np.min(self.F)
        F_median = np.median(self.F)
        F_absdiff = np.fabs(self.F1 - self.F)
        for i in range(self.popsize):
#            second_dim = np.full(7, -1.0)
            second_dim = np.full(8, np.nan)
            second_dim[0] = -1
            if self.F1[i] <= self.F[i]:
                second_dim[0] = self.opu[i]
                second_dim[1] = -self.F1[i]
                second_dim[2] = np.exp(-self.F1[i])
                second_dim[3] = self.F[i] - self.F1[i]
                if self.F1[i] <= F_min:
                    second_dim[4] = F_min - self.F1[i]
                
                if self.F1[i] <= self.best_so_far:
                    second_dim[5] = self.best_so_far - self.F1[i]
            
                if self.F1[i] <= F_median:
                        second_dim[6] = F_median - self.F1[i]
                
                second_dim[7] = (self.best_so_far / (self.F1[i] + 0.001)) * F_absdiff[i]
            
                self.window.append(self.opu[i], second_dim)
#                 # MUDITA: Loop for filling the window as improved offsping are generated
#                 # MANUEL: What is window? You sometimes index it like a list window[][] and other times like a matrix [, ]
#                 if np.any(self.window[:, 1] == np.inf):
#                     for value in range(self.max_window_size - 1, -1, -1):
#                         if self.window[value][0] == -1:
#                             self.window[value] = second_dim
#                             #print(self.window)
#                             break
#                 else:
#                     # MANUEL: What is this code doing? It is not obvious.
#                     # MUDITA: When a new sample comes in (second dimension), it finds from the last index a sample to replace that was generated by the same operator application. Once it gets the index of the sample to replace (nn), it removes it and shifts all samples one index below to put that new sample at top of window (index = 0). In case it could not find any existing application of that operator in the window, it replaces the worst candidate.
#                     for nn in range(self.max_window_size-1,-1,-1):
#                         if self.window[nn][0] == self.opu[i] or nn == 0:
#                             for nn1 in range(nn, 0, -1):
#                                 self.window[nn1] = self.window[nn1-1]
#                             self.window[0] = second_dim
#                             break
# #                       elif nn==0 and self.window[nn][0] != self.opu[i]:
#                             # MANUEL: What is window? You sometimes index it like a list window[][] and other times like a matrix [, ].
# #                           if self.F1[i] < np.max(self.window[: ,1]):
# #                               self.window[np.argmax(self.window[:,1])] = second_dim
            third_dim.append(second_dim)
        
        #print("Window in AOS update ", self.window, type(self.window), np.shape(self.window))
        
        self.gen_window.append(third_dim)
        #print("gen_window= ",type(self.gen_window), np.shape(self.gen_window), self.gen_window)
        #print(self.old_reward, self.reward)
        # MUDITA: Old_rewad and old_probability are communicating fine.
        self.old_reward = self.reward.copy()
        self.old_probability = self.probability.copy()
        # MUDITA: self.reward, self.quality and self.probability are changing as they change in the calc_.. definition.
        self.reward = self.reward_type.calc_reward()
        self.quality = self.quality_type.calc_quality(self.old_reward, self.reward, self.old_probability) #print("call to probability")
        self.probability = self.probability_type.calc_probability(self.quality)
        self.old_opu = self.opu

##################################################Other definitions######################################################################




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
    # Calculates Upper Confidence Bound as a quality
    ucb = reward + C * np.sqrt( 2 * np.log(np.sum(N)) / N)
    ucb[np.isinf(ucb) | np.isnan(ucb)] = 0
    return ucb

#def count_success(gen_window, i, j, off_met):
#    c_s = np.sum((gen_window[j, :, 0] == i) & (gen_window[j, :, off_met] != -1))
#    c_us = np.sum((gen_window[j, :, 0] == i) & (gen_window[j, :, off_met] == -1))
#    return c_s, c_us


# def angle_between(p1, p2):
#     # arctan2(y, x) computes the clockwise angle  (a value in radians between -pi and pi) between the origin and the point (x, y)
#     ang1 = np.arctan2(*p1[::-1]); # print("ang1", np.rad2deg(ang1))
#     ang2 = np.arctan2(*p2[::-1]); # print("ang2", np.rad2deg(ang2))
#     # second angle is subtracted from the first to get signed clockwise angular difference, that will be between -2pi and 2pi. Thus to get positive angle between 0 and 2pi, take the modulo against 2pi. Finally radians can be optionally converted to degrees using np.rad2deg.
#     #print("angle", np.rad2deg((ang1 - ang2) % (2 * np.pi)))
#     if (ang1 - ang2) % (2 * np.pi) > np.pi:
#         return 2 * np.pi - ((ang1 - ang2) % (2 * np.pi))
#     else:
#         return (ang1 - ang2) % (2 * np.pi)


#   def angle(vec, theta):
#       vec = vec * np.sign(vec[1])
#       angle = np.arccos(1 - distance.cosine(vec, np.array([1, 0]))) # In radian
#       return angle - np.deg2rad(theta)

##################################################Reward definitions######################################################################
    


def build_reward(choice, n_ops, rew_args, gen_window, window, off_met):
    if choice == 0:
        return Pareto_Dominance(n_ops, off_met, gen_window, rew_args["fix_appl"])
    elif choice == 1:
        return Pareto_Rank(n_ops, off_met, gen_window, rew_args["fix_appl"])
    elif choice == 2:
        return Compass_projection(n_ops, off_met, gen_window, rew_args["fix_appl"], rew_args["theta"])
    elif choice == 3:
        return Area_Under_The_Curve(n_ops, off_met, window, rew_args["window_size"], rew_args["decay"])
    elif choice == 4:
        return Sum_of_Rank(n_ops, off_met, window, rew_args["window_size"], rew_args["decay"])
    elif choice == 5:
        return Success_Rate(n_ops, off_met, gen_window, rew_args["max_gen"], rew_args["succ_lin_quad"], rew_args["frac"], rew_args["noise"])
    elif choice == 6:
        return Immediate_Success(n_ops, off_met, gen_window, rew_args["popsize"])
    elif choice == 7:
        return Success_sum(n_ops, off_met, gen_window, rew_args["max_gen"])
    elif choice == 8:
        return Normalised_success_sum_window(n_ops, off_met, window, rew_args["window_size"], rew_args["normal_factor"])
    elif choice == 9:
        return Normalised_success_sum_gen(n_ops, off_met, gen_window, rew_args["max_gen"])
    elif choice == 10:
        return Best2gen(n_ops, off_met, gen_window, rew_args["scaling_constant"], rew_args["alpha"], rew_args["alpha"])
    elif choice == 11:
        return Normalised_best_sum(n_ops, off_met, gen_window, rew_args["max_gen"], rew_args["intensity"], rew_args["alpha"])
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
    # TODO:
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
    # TODO:
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
    
    def __init__(self, n_ops, off_met, max_gen = None, window_size = None, decay = None, fix_appl = None):
        self.n_ops = n_ops
        # Offspring metric in range [1,7] stored in gen_window.
        self.off_met = off_met
        self.max_gen = int(max_gen)
        # MANUEL: So you have window_size here but no window?
        self.window_size = window_size
        self.fix_appl = fix_appl
        self.decay = decay
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
        # MUDITA: Old_reward is working fine
        self.old_reward = reward
        print("reward",reward)
        return reward

    def count_op_in_window(self, window):
        return count_op(self.n_ops, window, self.off_met) # print(window, window_op_sorted, N, rank)
    
    def count_total_succ_unsucc(self, n_ops, gen_window, j, off_met):
        """ Counts the number of successful and unsuccessful applications for each operator in generation 'j'"""
        total_success = np.zeros(n_ops)
        total_unsuccess = np.zeros(n_ops)
        for i in range(n_ops):
            if np.any(gen_window[j, :, 0] == i):
                # total_success[i], total_unsuccess[i] = count_success(gen_window, i, j, off_met)
                total_success[i] = np.sum((gen_window[j, :, 0] == i) & (gen_window[j, :, off_met] != np.nan))
                total_unsuccess[i] = np.sum((gen_window[j, :, 0] == i) & (gen_window[j, :, off_met] == np.nan))
        return total_success, total_unsuccess

    @abstractmethod
    def calc_reward(self):
        pass

def op_metric_for_fix_appl(gen_window, gen_window_len, op, fix_appl, off_met):
    b = []
    # Stop at fix_appl starting from the end of the window (latest fix_applications of operators)
    for j in range(gen_window_len-1, 0, -1):
        if np.any(gen_window[gen_window_len-1, :, 0] == i):
            value = gen_window[j, np.where((gen_window[j, :, 0] == i) & (gen_window[j, :, self.off_met] != np.nan)), self.off_met]
            b.append(value)
            if len(np.array(b).ravel()) == fix_appl:
                break
    b = np.array(b).ravel()
    return(b)

# MANUEL: These should have more descriptive names and a doctstring documenting
# where they come from (references) and what they do.
class Pareto_Dominance(RewardType):
    """
Jorge Maturana, Fr ́ed ́eric Lardeux, and Frederic Saubion. “Autonomousoperator management for evolutionary algorithms”. In:Journal of Heuris-tics16.6 (2010).https://link.springer.com/content/pdf/10.1007/s10732-010-9125-3.pdf, pp. 881–909.
"""

    def __init__(self, n_ops, off_met, gen_window, fix_appl = 20):
        super().__init__(n_ops, off_met, fix_appl = fix_appl)
        self.gen_window = gen_window
        debug_print("\n {} : fix_appl = {}".format(type(self).__name__, self.fix_appl))
    
    def calc_reward(self):
        # MANUEL: This function and the one for Pareto_Rank are almost identical! What's the difference?
        # MUDITA: Pareto dominance returns the number of operators dominated by an operator whereas Pareto rank gives the number of operators an operator is dominated by. Is there a library to calculate these values?
        reward = np.full(self.n_ops)
        std_op = np.full(self.n_ops, np.nan)
        mean_op = np.full(self.n_ops, np.nan)
        gen_window = self.gen_window
        gen_window_len = len(gen_window)
        # MANUEL: Why does it need to be converted to an array here?
        # MUDITA: In AOS_Update class, it is list because append works on list not array. So here I converted list to array.
        gen_window = np.array(gen_window)
        #print(type(gen_window), np.shape(gen_window), gen_window)
        for i in range(self.n_ops):
            b = op_metric_for_fix_appl(gen_window, gen_window_len, op, fix_appl, off_met)
             if len(b) > 0:
                std_op[i] = np.std(b)
                mean_op[i] = np.mean(b)
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
    def __init__(self, n_ops, off_met, gen_window, fix_appl = 20):
        super().__init__(n_ops, off_met, fix_appl = fix_appl)
        self.gen_window = gen_window
        debug_print("\n {} : fix_appl = {}".format(type(self).__name__, self.fix_appl))

    def calc_reward(self):
        reward = np.full(self.n_ops)
        std_op = np.full(self.n_ops, np.nan)
        mean_op = np.full(self.n_ops, np.nan)
        gen_window = self.gen_window
        gen_window = np.array(gen_window)
        gen_window_len = len(gen_window)
        for i in range(self.n_ops):
            b = op_metric_for_fix_appl(gen_window, gen_window_len, op, fix_appl, off_met)
            if len(b) > 0:
                std_op[i] = np.std(b)
                mean_op[i] = np.mean(b)
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
        #print(reward)
        if np.sum(reward) != 0:
            reward /= np.sum(reward)
        reward = 1. - reward
        return super().check_reward(reward)


class Compass_projection(RewardType):
    """
        Jorge Maturana and Fr ́ed ́eric Saubion. “A compass to guide genetic al-gorithms”. In:International Conference on Parallel Problem Solving fromNature.http://www.info.univ-angers.fr/pub/maturana/files/MaturanaSaubion-Compass-PPSNX.pdf. Springer. 2008, pp. 256–265.
        """
    def __init__(self, n_ops, off_met, gen_window, fix_appl = 100, theta = 45):
        super().__init__(n_ops, off_met, fix_appl = fix_appl)
        self.gen_window = gen_window
        self.theta = theta
        debug_print("\n {} : fix_appl = {}".format(type(self).__name__, self.fix_appl, self.theta))
    
    def calc_reward(self):
        gen_window = self.gen_window
        # MANUEL: This should be an array alreadY?
        # MUDITA: No, in AOS_Update class, it is list because append works on list not array. So here I converted list to array.
        gen_window = np.array(gen_window)
        gen_window_len = len(gen_window)
        reward = np.zeros(self.n_ops)
        # Projection on line B with thetha = pi/4
#        B = [1, 1]
        for i in range(self.n_ops):
            b = op_metric_for_fix_appl(gen_window, gen_window_len, op, fix_appl, off_met)
            
            if len(b) > 0:
                # Diversity
                std = np.std(b)
                # Quality 
                avg = np.mean(b)
                # abs(atan(Quality / Diversity) - theta)
                angle = np.fabs(np.arctan(np.deg2rad(avg / std)) - self.theta)
                # Euclidean distance of the vector 
                reward[i] = (np.sqrt(std**2 + avg**2)) * np.cos(angle)
                # Maturana & Sablon (2008) divide by T_it defined as mean execution time of operator i over its last t applications.
                # We do not divide here.
        reward = reward - np.min(reward)
        return super().check_reward(reward)

class Area_Under_The_Curve(RewardType):
    """
Alvaro Fialho, Marc Schoenauer, and Mich`ele Sebag. “Toward comparison-based adaptive operator selection”. In:Proceedings of the 12th annual con-ference on Genetic and evolutionary computation.https://hal.inria.fr/file/index/docid/471264/filename/banditGECCO10.pdf. ACM.2010, pp. 767–774
"""
    def __init__(self, n_ops, off_met, window, window_size = 50, decay = 0.4):
        super().__init__(n_ops, off_met, window_size = window_size, decay = decay)
        # MANUEL: This window is not the same object as the one in AOS!
        self.window = window
        self.window_size = window_size
        #print("Window in AUC init ", self.window, type(self.window), np.shape(self.window))
        debug_print("\n {} : window_size = {}, decay = {}".format(type(self).__name__, self.window_size, self.decay))
    
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
    def __init__(self, n_ops, off_met, window, window_size = 50, decay = 0.4):
        super().__init__(n_ops, off_met, window_size = window_size, decay = decay)
        self.window = window
        self.window_size = window_size
        debug_print("\n {} : window_size = {}, decay = {}".format(type(self).__name__, self.window_size, self.decay))
    
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

# MANUEL: Create a new class SuccessRateType. That should contain the gen_window and this common function:
# MUDITA: I have added this function in abstract class RewardType.
#def count_total_succ_unsucc(n_ops, gen_window, j, off_met):
#    total_success = np.zeros(n_ops)
#    total_unsuccess = np.zeros(n_ops)
#    for i in range(n_ops):
        # MANUEL: What is gen_window? Is it a matrix, a list, a cube?
        # MUDITA: It is a 3-D list.
#        if np.any(gen_window[j, :, 0] == i):
#            total_success[i], total_unsuccess[i] = count_success(gen_window, i, j, off_met)
#    return total_success, total_unsuccess


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
    
    def __init__(self, n_ops, off_met, gen_window, max_gen = 10, succ_lin_quad = 1, frac = 0.01, noise = 0.0):
        super().__init__(n_ops, off_met, max_gen = max_gen)
        self.gen_window = gen_window
        self.succ_lin_quad = succ_lin_quad
        self.frac = frac
        self.noise = noise
        debug_print("\n {} : max_gen = {}, succ_lin_quad = {}, frac = {}, noise = {}".format(type(self).__name__, self.max_gen, self.succ_lin_quad, self.frac, self.noise))
    
    def calc_reward(self):
        gen_window = self.gen_window
        # MANUEL: Should be an array already?
        # MUDITA: No, in AOS_Update class, it is list because append works on list not array. So here I converted list to array.
        gen_window = np.array(gen_window)
        gen_window_len = len(gen_window)
        max_gen = self.max_gen
        if gen_window_len < max_gen:
            max_gen = gen_window_len
        reward = np.zeros(self.n_ops)
        for j in range(gen_window_len - max_gen, gen_window_len):
            total_success, total_unsuccess = super().count_total_succ_unsucc(n_ops, gen_window, j, off_met)
            napplications = total_success + total_unsuccess
            # Avoid division by zero. If total == 0, then total_success is zero.
            napplications[napplications == 0] = 1
            reward += (total_success ** self.succ_lin_quad + self.frac * np.sum(total_success)) / napplications
        reward += self.noise
        return super().check_reward(reward)


class Immediate_Success(RewardType):
    """
 Mudita  Sharma,  Manuel  L ́opez-Ib ́a ̃nez,  and  Dimitar  Kazakov.  “Perfor-mance Assessment of Recursive Probability Matching for Adaptive Oper-ator Selection in Differential Evolution”. In:International Conference onParallel Problem Solving from Nature.http://eprints.whiterose.ac.uk/135483/1/paper_66_1_.pdf. Springer. 2018, pp. 321–333.
 """
    def __init__(self, n_ops, off_met, gen_window, popsize):
        super().__init__(n_ops, off_met)
        self.gen_window = gen_window
        self.popsize = popsize
    
    def calc_reward(self):
        gen_window = self.gen_window
        gen_window = np.array(gen_window)
        total_success, total_unsuccess = super().count_total_succ_unsucc(self.n_ops, gen_window, len(gen_window) - 1, self.off_met)
        reward = total_success / self.popsize
        return super().check_reward(reward)

class Success_sum(RewardType):
    """
 Christian  Igel  and  Martin  Kreutz.  “Operator  adaptation  in  evolution-ary  computation  and  its  application  to  structure  optimization  of  neu-ral  networks”.  In:Neurocomputing55.1-2  (2003).https : / / ac . els -cdn.com/S0925231202006288/1-s2.0-S0925231202006288-main.pdf?_tid=c6274e78-02dc-4bf6-8d92-573ce0bed4c4&acdnat=1540907096_d0cc1e2b4ca56a49587b4d55e1008a84, pp. 347–361.
 """
    def __init__(self, n_ops, off_met, gen_window, max_gen = 4):
        super().__init__(n_ops, off_met, max_gen = max_gen)
        self.gen_window = gen_window
        debug_print("\n {} : max_gen = {}".format(type(self).__name__, self.max_gen))
    
    def calc_reward(self):
        gen_window = self.gen_window
        gen_window = np.array(gen_window)
        gen_window_len = len(gen_window)
        max_gen = self.max_gen
        if gen_window_len < max_gen:
            max_gen = gen_window_len
        napplications = np.zeros(self.n_ops)
        reward = np.zeros(self.n_ops)
        for j in range(gen_window_len - max_gen, gen_window_len):
            total_success, total_unsuccess = super().count_total_succ_unsucc(self.n_ops, gen_window, j, self.off_met)
            napplications += total_success + total_unsuccess
            for i in range(self.n_ops):
                reward[i] += np.sum(gen_window[j, np.where((gen_window[j, :, 0] == i) & (gen_window[j, :, self.off_met] != np.nan)), self.off_met])
        napplications[napplications == 0] = 1
        reward /= napplications
        return super().check_reward(reward)

class Normalised_success_sum_window(RewardType):
    """
Alvaro Fialho, Marc Schoenauer, and Mich`ele Sebag. “Analysis of adaptiveoperator selection techniques on the royal road and long k-path problems”.In:Proceedings of the 11th Annual conference on Genetic and evolutionarycomputation.https://hal.archives-ouvertes.fr/docs/00/37/74/49/PDF/banditGECCO09.pdf. ACM. 2009, pp. 779–786.
"""
    def __init__(self, n_ops, off_met, window, window_size = 50, normal_factor = 0.1):
        super().__init__(n_ops, off_met, window_size = window_size)
        self.window = window
        self.window_size = window_size
        self.normal_factor = normal_factor
        debug_print("\n {} : window_size = {}, normal_factor = {}".format(type(self).__name__, self.window_size, self.normal_factor))
    
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
    def __init__(self, n_ops, off_met, gen_window, max_gen = 4):
        super().__init__(n_ops, off_met, max_gen = max_gen)
        self.gen_window = gen_window
        debug_print("\n {} : max_gen = {}".format(type(self).__name__, self.max_gen))
    
    def calc_reward(self):
        gen_window = self.gen_window
        gen_window = np.array(gen_window)
        gen_window_len = len(gen_window)
        max_gen = self.max_gen
        if gen_window_len < max_gen:
            max_gen = gen_window_len
        reward = np.zeros(self.n_ops)
        for j in range(gen_window_len - max_gen, gen_window_len):
            # PLease use count_total_succ_and_unsucc()
            #total_success = 0; total_unsuccess = 0
            total_success, total_unsuccess = super().count_total_succ_unsucc(self.n_ops, gen_window, j, self.off_met)
            napplications = total_success + total_unsuccess
            napplications[napplications == 0] = 1
            value = np.zeros(self.n_ops)
            for i in range(self.n_ops):
                value[i] = np.sum(gen_window[j, np.where((gen_window[j,:,0] == i) & np.logical_not(np.isnan(gen_window[j, :, self.off_met]))) , self.off_met])
            reward += value / napplications
            
        return super().check_reward(reward)

class Best2gen(RewardType):
    """
Giorgos Karafotias, Agoston Endre Eiben, and Mark Hoogendoorn. “Genericparameter  control  with  reinforcement  learning”.  In:Proceedings of the2014 Annual Conference on Genetic and Evolutionary Computation.http://www.few.vu.nl/~gks290/papers/GECCO2014-RLControl.pdf. ACM.2014, pp. 1319–1326.
 """
    def __init__(self, n_ops, off_met, gen_window, scaling_constant = 1, alpha = 0, beta = 1):
        super().__init__(n_ops, off_met)
        self.gen_window = gen_window
        self.scaling_constant = scaling_constant
        self.alpha = alpha
        self.beta = beta
        debug_print("\n {} : scaling constant = {}, alpha = {}, beta = {}".format(type(self).__name__, self.scaling_constant, self.alpha, self.beta))
    
    def calc_reward(self):
        # Involves calculation of best in previous two generations.
        gen_window = self.gen_window
        gen_window = np.array(gen_window)
        gen_window_len = len(gen_window)
        total_success_t, total_unsuccess_t = super().count_total_succ_unsucc(self.n_ops, gen_window, gen_window_len - 1, off_met)
        if gen_window_len >= 2:
            total_success_t_1, total_unsuccess_t_1 = super().count_total_succ_unsucc(self.n_ops, gen_window, gen_window_len - 2, off_met)
        else:
            total_success_t_1 = 0
            total_unsuccess_t_1 = 0
        n_applications = (total_success_t + total_unsuccess_t) - (total_success_t_1 + total_unsuccess_t_1)

        best_t = np.zeros(self.n_ops)
        best_t_1 = np.zeros(self.n_ops)
        for i in range(self.n_ops):
            # Calculating best in current generation
            if np.any(np.logical_not(np.isnan(gen_window[gen_window_len-1, :, self.off_met]))):
                best_t[i] = np.max(gen_window[gen_window_len-1, np.where((gen_window[gen_window_len-1, :, 0] == i) & (gen_window[gen_window_len-1, :, self.off_met] != np.nan)), self.off_met])
            # Calculating best in last generation
            if gen_window_len >= 2 and np.any(np.logical_not(np.isnan(gen_window[gen_window_len-2, :, self.off_met]))):
                best_t_1[i] = np.max(gen_window[gen_window_len-2, np.where((gen_window[gen_window_len-2, :, 0] == i) & (gen_window[gen_window_len-2, :, self.off_met] != np.nan)), self.off_met])
        best_t_1[best_t_1 == 0] = 1
        n_applications[n_applications == 0] = 1
        reward = self.scaling_constant * np.fabs(best_t - best_t_1) / ((best_t_1**self.alpha) * (np.fabs(n_applications)**self.beta))
        return super().check_reward(reward)

class Normalised_best_sum(RewardType):
    """
Alvaro Fialho, Marc Schoenauer, and Mich`ele Sebag. “Analysis of adaptiveoperator selection techniques on the royal road and long k-path problems”.In:Proceedings of the 11th Annual conference on Genetic and evolutionarycomputation.https://hal.archives-ouvertes.fr/docs/00/37/74/49/PDF/banditGECCO09.pdf. ACM. 2009, pp. 779–786.
"""
    def __init__(self, n_ops, off_met, gen_window, max_gen = 10, intensity = 0, alpha = 1):
        super().__init__(n_ops, off_met, max_gen = max_gen)
        self.gen_window = gen_window
        self.intensity = intensity
        self.alpha = alpha
        debug_print("\n {} : max_gen = {}, intensity = {}, alpha = {}".format(type(self).__name__, self.max_gen, self.intensity, self.alpha))
    
    def calc_reward(self):
        # Normalised best sum
        reward = np.zeros(self.n_ops)
        gen_window = self.gen_window
        gen_window = np.array(gen_window)
        gen_window_len = len(gen_window)
        max_gen = self.max_gen
        if gen_window_len < max_gen:
            max_gen = gen_window_len
        for i in range(self.n_ops):
            # list of best metric value produce by operator i at each generation.
            for j in range(gen_window_len - max_gen, gen_window_len):
                # MANUEL: Use count_total_succ_unsucc()
                # MUDITA: We donot use this information (number of applications of an operator) here.
                if np.any((gen_window[j,:,0] == i) & (gen_window[j, :, self.off_met] != np.nan)):
                    # -1 means unsuccess, it should np.nan
#                    reward[i] += (np.max(np.hstack(gen_window[j, np.where((gen_window[j,:,0] == i) & (gen_window[j, :, self.off_met] != np.nan)), self.off_met])))
                    reward[i] += np.max(gen_window[j, np.where((gen_window[j,:,0] == i) & (gen_window[j, :, self.off_met] != np.nan)), self.off_met])
        reward = (1.0 / max_gen) * (reward**self.intensity) / (np.max(reward)**self.alpha)
        return super().check_reward(reward)


##################################################Quality definitions######################################################################

def build_quality(choice, n_ops, qual_args, window, off_met):
    if choice == 0:
        return Weighted_sum(n_ops, qual_args["decay_rate"])
    elif choice == 1:
        return Upper_confidence_bound(n_ops, off_met, window, qual_args["scaling_factor"])
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
        debug_print("{}: quality: {}".format(type(self).__name__, quality))
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
        debug_print("\n {} : decay_rate = {}".format(type(self).__name__, self.decay_rate))
    
    def calc_quality(self, old_reward, reward, old_probability):
        quality = self.decay_rate * reward + (1.0 - self.decay_rate) * self.old_quality
        return super().check_quality(quality)

class Upper_confidence_bound(QualityType):
    """
Alvaro Fialho et al. “Extreme value based adaptive operator selection”.In:International Conference on Parallel Problem Solving from Nature.https : / / hal . inria . fr / file / index / docid / 287355 / filename /rewardPPSN.pdf. Springer. 2008, pp. 175–184
"""
    def __init__(self, n_ops, off_met, window, scaling_factor = 0.5):
        super().__init__(n_ops)
        self.off_met = off_met
        self.window = window
        self.scaling_factor = scaling_factor
        debug_print("\n {} : scaling_factor = {}".format(type(self).__name__, self.scaling_factor))
    
    def calc_quality(self, old_reward, reward, old_probability):
        #window_op_sorted, N, rank = count_op(self.n_ops, self.window, self.off_met)
        N = self.window.count_ops()
        quality = UCB(N, self.scaling_factor, reward)
        return super().check_quality(quality)

class Identity(QualityType):
    def __init__(self, n_ops):
        super().__init__(n_ops)
    
    def calc_quality(self, old_reward, reward, old_probability):
        quality[:] = reward[:]
        #print("In definition",quality)
        return super().check_quality(quality)

class Weighted_normalised_sum(QualityType):
    """
Christian  Igel  and  Martin  Kreutz.  “Operator  adaptation  in  evolution-ary  computation  and  its  application  to  structure  optimization  of  neu-ral  networks”.  In:Neurocomputing55.1-2  (2003).https : / / ac . els -cdn.com/S0925231202006288/1-s2.0-S0925231202006288-main.pdf?_tid=c6274e78-02dc-4bf6-8d92-573ce0bed4c4&acdnat=1540907096_d0cc1e2b4ca56a49587b4d55e1008a84, pp. 347–361
"""
    def __init__(self, n_ops, decay_rate = 0.3):
        super().__init__(n_ops)
        self.decay_rate = decay_rate
        debug_print("\n {} : decay_rate = {}".format(type(self).__name__, self.decay_rate))
    
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
        debug_print("\n {} : weight_reward = {}, weight_old_reward = {}, discount_rate = {}".format(type(self).__name__, self.weight_reward, self.weight_old_reward, self.discount_rate))
    
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
        assert self.p_min != 1.0 / len(probability)
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
        debug_print("check_probability(): probability: ", probability, "\n")
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
        debug_print("\n {} : p_min = {}, error_prob = {}".format(type(self).__name__, self.p_min, self.error_prob))
        
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
        debug_print("\n {} : p_min = {}, p_max = {}, learning_rate = {}".format(type(self).__name__, self.p_min, self.p_max, self.learning_rate))

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
        debug_print("\n {} : p_min = {}, learning_rate = {}".format(type(self).__name__, self.p_min, self.learning_rate))
        
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

