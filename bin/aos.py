from __future__ import print_function
import numpy as np
from scipy.stats import rankdata
import math
from collections import Counter
from scipy.spatial import distance

import sys
def debug_print(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
# MANUEL: What is the difference between AOS and unknown AOS?
class AOS(object):
    def __init__(self, popsize, F1, F, u, X, f_min, x_min, best_so_far, best_so_far1, n_ops, window_size= None):
        self.popsize = int(popsize)
        self.F1 = F1
        self.F = F
        self.u = u
        self.X = X
        self.f_min = f_min
        self.x_min = x_min
        self.best_so_far = best_so_far
        self.best_so_far1 = best_so_far1
        self.n_ops = n_ops
        self.window_size = window_size
        
        # The initial list of operators (randomly permuted)
        self.op_init_list = list(np.random.permutation(n_ops))
        self.opu = [4 for i in range(int(popsize))]; self.opu = np.array(self.opu)
        self.old_opu = self.opu; self.old_opu = np.array(self.old_opu)
        self.number_metric = 7
        self.window = [[np.inf for j in range(self.number_metric)] for i in range(self.window_size)]; self.window = np.array(self.window); self.window[:, 0].fill(-1)
        
        self.gen_window = []; # print("Inside AOS", type(self.gen_window), self.gen_window)
        self.total_success = []
        self.total_unsuccess = []
    
        self.area = np.zeros(int(self.n_ops))
    
    
    ##################################################Offspring Metric definitions##################################################################
    def OM_Update(self):
        third_dim = []
        # success = np.zeros(self.n_ops); unsuccess = np.zeros(self.n_ops)
        for i in range(self.popsize):
            second_dim = np.zeros(7);
            if self.F1[i] <= self.F[i]:
                # success[self.opu[i]] += 1
                second_dim[0] = self.opu[i]
                second_dim[1] = np.exp(-self.F1[i])
                second_dim[2] = self.F[i] - self.F1[i]
                if self.F1[i] <= np.min(self.F):
                    second_dim[3] = np.min(self.F) - self.F1[i]
                else:
                    second_dim[3] = -1
                if self.F1[i] <= self.best_so_far:
                    second_dim[4] = self.best_so_far - self.F1[i]
                else:
                    second_dim[4] = -1
                if self.F1[i] <= np.median(self.F):
                    second_dim[5] = np.median(self.F) - self.F1[i]
                else:
                    second_dim[5] = -1
                second_dim[6] = (self.best_so_far / (self.F1[i]+0.001)) * math.fabs(self.F1[i] - self.F[i])
                
                if np.any(self.window[:, 1] == np.inf):
                    for value in range(self.window_size-1,-1,-1):
                        if self.window[value][0] == -1:
                            self.window[value] = second_dim
                            #print(self.window)
                            break
                else:
                    for nn in range(self.window_size-1,-1,-1):
                        if self.window[nn][0] == self.opu[i]:
                            for nn1 in range(nn, 0, -1):
                                self.window[nn1] = self.window[nn1-1]
                            self.window[0] = second_dim
                            break
                        elif nn==0 and self.window[nn][0] != self.opu[i]:
                            if self.F1[i] < np.max(self.window[: ,1]):
                                self.window[np.argmax(self.window[:,1])] = second_dim
                #self.X[i][:] = self.u[i][:]
                #self.F[i] = self.F1[i]; print("aos", self.F)
                third_dim.append(second_dim);# print("r, rule: ",r, rule)
            else:
                # unsuccess[self.opu[i]] += 1
                second_dim = [-1 for i in range(7)]
                second_dim[0] = self.opu[i]
                third_dim.append(second_dim)
        
        #print("Outside ", self.window)
        self.gen_window.append(third_dim); # print("gen_window= ",self.gen_window, type(self.gen_window), np.shape(self.gen_window))
        # self.gen_window = np.array(self.gen_window)
        # self.total_success.append(success); self.total_unsuccess.append(unsuccess); #print("Su",success); print("Un",unsuccess); print("TSu",self.total_success); print("TUn",self.total_unsuccess);
        #print("call to reward")
        self.Reward(); #print("call to quality")
        self.Quality(); #print("call to probability")
        self.probability = self.probability_type.calc_probability(self.quality)
        self.old_opu = self.opu

##################################################Other definitions######################################################################

# Return sorted window, number of successful applications of operators and rank

def count_op(n_ops, window, Off_met): # ???????Include ranking for minimising case??????? Use W-r in place r
    # Gives rank to window[:, Off_met]: largest number will get largest number rank
    rank = rankdata(window[:, Off_met].round(1), method = 'min')
    order = rank.argsort()
    # order gives the index of rank in ascending order. Sort operators and rank in increasing rank.
    window_op_sorted = window[order, 0];
    rank = rank[order]
    rank = rank[window_op_sorted >= 0]
    window_op_sorted = window_op_sorted[window_op_sorted >= 0]; # print("window_op_sorted = ",window, window_op_sorted, rank, order)

    # counts number of times an operator is present in the window
    N = np.zeros(n_ops); N = np.array(N); #print(N, window_op_sorted)
    # the number of times each operator appears in the sliding window
    op, count = np.unique(window_op_sorted, return_counts=True); # print(op, count)
    for i in range(len(count)):
        # print(len(count), i, op[i], count[i])
        N[op[i]] = count[i]
    #N[op] = count
    return window_op_sorted, N, rank

# Count the successful number of applications in fix number of generations

#def count_unsuccessful_applications(gen_window):
    #x, y = np.unique(gen_window[:,:,0], return_counts = True)
    #if np.any(x, -1):
        #for i in range(len(x)):
            #if x[i] == -1:
                #return y[i]
    #else:
        #return 0

# Calculates Transitive Matrix

def TM(n_ops, p):
    ## MANUEL: I think this loop can be replaced by
    # tran_matrix = p + p[, np.newaxis]
    ## search and read about Numpy broadcasting.
    tran_matrix = np.zeros((n_ops, n_ops))
    for i in range(n_ops):
        for j in range(n_ops):
            tran_matrix[i][j] = p[j] + p[i]; # print(tran_matrix[i][j])
    tran_matrix = normalize_matrix(tran_matrix)
    return tran_matrix

# Normalise n_ops dimensional matrix

def normalize_matrix(x):
    return x / np.sum(x, axis=1)[:, None]

def calc_delta_r (decay_reward3, W, ndcg):
    if decay_reward3 == 0:
        return np.ones(window_size)
    r = np.array(range(W), dtype='float')
    if ndcg:
        r += 1
        delta_r = ((2 ** (W - r)) - 1) / np.log(1 + r)
    else:
        delta_r = (decay_reward3 ** r) * (W - r)
    return delta_r

## MANUEL: Instead of adding comments before functions, add docstrings after
## the function def. This way one can read the comment by doing ?AUC in Python
def AUC(operators, rank, op, decay_reward3, ndcg = True):
    """Calculates area under the curve for each operator"""
    assert len(operators) == len(rank)
    W = len(operators)
    delta_r_vector = calc_delta_r(decay_reward3, W, ndcg)
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

# Calculates Upper Confidence Bound as a quality

def upper_confidence_bound (N, C, reward):
    ucb = reward + C * np.sqrt( 2 * np.log(np.sum(N))/(N))
    # Infinite quality means division by zero, give zero quality.
    ucb[np.isinf(ucb)] = 0
    return ucb

def count_success(popsize, gen_window, i, j, Off_met):
    c_s = 0; c_us = 0
    for k in range(popsize):
        if gen_window[j, k, 0] == i and gen_window[j, k, Off_met] != -1:
            c_s += 1
        if gen_window[j, k, 0] == i and gen_window[j, k, Off_met] == -1:
            c_us += 1
    return c_s, c_us


'''
def angle_between(p1, p2):
    # arctan2(y, x) computes the clockwise angle  (a value in radians between -pi and pi) between the origin and the point (x, y)
    ang1 = np.arctan2(*p1[::-1]); # print("ang1", np.rad2deg(ang1))
    ang2 = np.arctan2(*p2[::-1]); # print("ang2", np.rad2deg(ang2))
    # second angle is subtracted from the first to get signed clockwise angular difference, that will be between -2pi and 2pi. Thus to get positive angle between 0 and 2pi, take the modulo against 2pi. Finally radians can be optionally converted to degrees using np.rad2deg.
    #print("angle", np.rad2deg((ang1 - ang2) % (2 * np.pi)))
    if (ang1 - ang2) % (2 * np.pi) > np.pi:
        return 2 * np.pi - ((ang1 - ang2) % (2 * np.pi))
    else:
        return (ang1 - ang2) % (2 * np.pi)
'''


##################################################Reward definitions######################################################################

                                                        ##########################Diversity-Quality based#################################

# index: 0 Applicable for fix number of generation
# Parameter(s): max_gen
def Reward0(popsize, n_ops, window_size, window, gen_window, Off_met, max_gen, decay_reward3, decay_reward4, int_a_reward5, b_reward5, e_reward5, a_reward71, c_reward9, int_b_reward9, int_a_reward9, int_a_reward101, b_reward101, opu, old_opu, total_success, total_unsuccess, old_reward):
    reward = np.zeros(n_ops)
    s_op = np.zeros(n_ops)
    q_op = np.zeros(n_ops)
    gen_window = np.array(gen_window)
    if max_gen > len(gen_window):
        max_gen = len(gen_window)
    for i in range(n_ops):
        b = []
        for j in range(len(gen_window)-1, len(gen_window)-max_gen-1, -1):
            #print("GW........",gen_window[j])
            if np.any(gen_window[j, :, 0] == i):
                b.append(gen_window[j, np.where(gen_window[j, :, 0] == i) and np.where(gen_window[j, :, Off_met] != -1), Off_met])
        #print(b)
        if b != []:
            s_op[i] = np.std(np.hstack(b)); q_op[i] = np.average(np.hstack(b))
        else:
            s_op[i] = 0; q_op[i] = 0
        #print("b", b)
    #a.append(b)
    #print("a", s_op, q_op)
    for i in range(n_ops):
        for j in range(n_ops):
            if i != j:
                #print(gen_window[len(gen_window)-1], i, j)
                if s_op[i] != [] and s_op[j] != []:
                    #print("D, Q:  ",A1, A2, op_Diversity(gen_window, j, Off_met), op_Quality(gen_window, j, Off_met))
                    if s_op[i] > s_op[j] and q_op[i] > q_op[j]:
                        reward[i] += 1
    #print(reward)
        if np.sum(reward) != 0:
            reward = (reward / np.sum(reward))
    return reward


# index: 1 Applicable for fix number of generation
# Parameter(s): max_gen
def Reward1(popsize, n_ops, window_size, window, gen_window, Off_met, max_gen, decay_reward3, decay_reward4, int_a_reward5, b_reward5, e_reward5, a_reward71, c_reward9, int_b_reward9, int_a_reward9, int_a_reward101, b_reward101, opu, old_opu, total_success, total_unsuccess, old_reward):
    reward = np.zeros(n_ops)
    s_op = np.zeros(n_ops)
    q_op = np.zeros(n_ops)
    gen_window = np.array(gen_window)
    if max_gen > len(gen_window):
        max_gen = len(gen_window)
    for i in range(n_ops):
        b = []
        for j in range(len(gen_window)-1, len(gen_window)-max_gen-1, -1):
            #print("GW........",gen_window[j])
            if np.any(gen_window[j, :, 0] == i):
                b.append(gen_window[j, np.where(gen_window[j, :, 0] == i) and np.where(gen_window[j, :, Off_met] != -1), Off_met])
        #print(b)
        if b != []:
            s_op[i] = np.std(np.hstack(b)); q_op[i] = np.average(np.hstack(b))
        else:
            s_op[i] = 0; q_op[i] = 0
        #print("b", b)
    #a.append(b)
    #print("a", s_op, q_op)
    for i in range(n_ops):
        for j in range(n_ops):
            if i != j:
                #print(gen_window[len(gen_window)-1], i, j)
                if s_op[i] != [] and s_op[j] != []:
                    #print("D, Q:  ",A1, A2, op_Diversity(gen_window, j, Off_met), op_Quality(gen_window, j, Off_met))
                    if s_op[i] < s_op[j] and q_op[i] < q_op[j]:
                        reward[i] += 1
    #print(reward)
    if np.sum(reward) != 0:
        reward = (reward / np.sum(reward))
    return -reward


# index: 2 Applicable for current generation
# Parameter(s): max_gen
def Reward2(popsize, n_ops, window_size, window, gen_window, Off_met, max_gen, decay_reward3, decay_reward4, int_a_reward5, b_reward5, e_reward5, a_reward71, c_reward9, int_b_reward9, int_a_reward9,int_a_reward101, b_reward101, opu, old_opu, total_success, total_unsuccess, old_reward):
    reward = np.zeros(n_ops)
    gen_window = np.array(gen_window)
    if max_gen > len(gen_window):
        max_gen = len(gen_window)
    # Projection on line B with thetha = pi/4
    B = [1, 1]
    for i in range(n_ops):
        b = []
        for j in range(len(gen_window)-1, len(gen_window)-max_gen-1, -1):
            if np.any(gen_window[len(gen_window)-1, :, 0] == i):
                b.append(gen_window[j, np.where(gen_window[j, :, 0] == i) and np.where(gen_window[j, :, Off_met] != -1), Off_met])
        if b != []:
            # Diversity: np.std(np.hstack(b)) and Quality: np.average(np.hstack(b))
            reward[i] = 1-distance.cosine([np.std(np.hstack(b)), np.average(np.hstack(b))], B); # print(b, reward[i])
    reward = reward - np.min(reward)
    return reward

                                                        ##########################Comparison (Rank) based########################################

# index: 3 Applicable for fix size window
# Parameter(s): window, decay_reward3
def Reward3(popsize, n_ops, window_size, window, gen_window, Off_met, max_gen, decay_reward3, decay_reward4, int_a_reward5, b_reward5, e_reward5, a_reward71, c_reward9, int_b_reward9, int_a_reward9,int_a_reward101, b_reward101, opu, old_opu, total_success, total_unsuccess, old_reward):
    reward = np.zeros(n_ops)
    window = window[window[:, Off_met] != -1][:, :];
    window_op_sorted, N, rank = count_op(n_ops, window, Off_met); # print(window, window_op_sorted, N, rank)
    for op in range(n_ops):
        reward[op] = AUC(window_op_sorted, rank, op, decay_reward3)
    # print("Inside reward: ", reward)
    return reward

# index: 4 Applicable for fix size window (Done!)
# Parameter(s): window, decay_reward4
def Reward4(popsize, n_ops, window_size, window, gen_window, Off_met, max_gen, decay_reward3, decay_reward4, int_a_reward5, b_reward5, e_reward5, a_reward71, c_reward9, int_b_reward9, int_a_reward9,int_a_reward101, b_reward101, opu, old_opu, total_success, total_unsuccess, old_reward):
    reward = np.zeros(n_ops)
    window = window[window[:, Off_met] != -1][:, :];
    window_op_sorted, N, rank = count_op(n_ops, window, Off_met)
    for i in range(len(window_op_sorted)):
        value = (decay_reward4 ** rank[i]) * (window_size - rank[i])
        ## MANUEL: If window_op_sorted only contains values from 0 to n_ops, then this loop can be simply:
        # reward[window_op_sorted] += value
        ## Test it!
        for j in range(n_ops):
            if window_op_sorted[i] == j:
                reward[j] += value
    reward /= np.sum(reward)
    return reward

########################Success based###########################################

# index: 5 Applicable for fix number of generations
# Parameter(s): max_gen, int_a_reward5, b_reward5, e_reward5
def Reward5(popsize, n_ops, window_size, window, gen_window, Off_met, max_gen, decay_reward3, decay_reward4, int_a_reward5, b_reward5, e_reward5, a_reward71, c_reward9, int_b_reward9, int_a_reward9, int_a_reward101, b_reward101, opu, old_opu, total_success, total_unsuccess, old_reward):
    reward = np.zeros(n_ops)
    gen_window = np.array(gen_window)
    if len(gen_window) < max_gen:
        max_gen = len(gen_window)
    for j in range(len(gen_window)-1, len(gen_window)-max_gen-1, -1):
        total_success = np.zeros(n_ops); total_unsuccess = np.zeros(n_ops)
        for i in range(n_ops):
            if np.any(gen_window[j, :, 0] == i):
                total_success[i], total_unsuccess[i] = count_success(popsize, gen_window, i, j, Off_met)
        #print(total_success, total_unsuccess, np.sum(total_success))
        for i in range(n_ops):
            if total_success[i] + total_unsuccess[i] != 0:
                reward[i] += (total_success[i]**int_a_reward5 + b_reward5 * np.sum(total_success)) / (total_success[i] + total_unsuccess[i])
            else:
                reward[i] += 0
    reward += e_reward5
    return reward

# index: 6 Applicable for last generation
# Parameter(s): None
def Reward6(popsize, n_ops, window_size, window, gen_window, Off_met, max_gen, decay_reward3, decay_reward4, int_a_reward5, b_reward5, e_reward5, a_reward71, c_reward9, int_b_reward9, int_a_reward9, int_a_reward101, b_reward101, opu, old_opu, total_success, total_unsuccess, old_reward):
    reward = np.zeros(n_ops)
    gen_window = np.array(gen_window)
    
    for i in range(n_ops):
        total_success = 0; total_unsuccess = 0
        if np.any(gen_window[len(gen_window)-1, :, 0] == i):
            total_success, total_unsuccess = count_success(popsize, gen_window, i, len(gen_window)-1, Off_met); #print(total_unsuccess, total_success)
            reward[i] = (np.array(total_success) / popsize)
    return reward

                                                        ##########################Weighted offspring based################################

# index: 7 Applicable for fix number of generations
# Parameter(s): max_gen
def Reward70(popsize, n_ops, window_size, window, gen_window, Off_met, max_gen, decay_reward3, decay_reward4, int_a_reward5, b_reward5, e_reward5, a_reward71, c_reward9, int_b_reward9, int_a_reward9, int_a_reward101, b_reward101, opu, old_opu, total_success, total_unsuccess, old_reward):
    reward = np.zeros(n_ops)
    gen_window = np.array(gen_window)
    appl = np.zeros(n_ops)
    if len(gen_window) < max_gen:
        max_gen = len(gen_window)
    for i in range(n_ops):
        appl = 0
        for j in range(len(gen_window)-1, len(gen_window)-max_gen-1, -1):
            total_success = 0; total_unsuccess = 0
            #print("first: ",gen_window, np.shape(gen_window), len(gen_window), i, j);print(gen_window[j,0,0])
            if np.any(gen_window[j, :, 0] == i):
                total_success, total_unsuccess = count_success(popsize, gen_window, i, j, Off_met)
                reward[i] += np.sum(gen_window[j, np.where(gen_window[j, :, 0] == i) and np.where(gen_window[j, :, Off_met] != -1), Off_met])
                appl += total_success + total_unsuccess
        if appl != 0:
            reward[i] = np.array(reward[i]) / (np.array(appl))
        else:
            reward[i] = 0
    return reward


# index: 8 Applicable for fix window size
# Parameter(s): window, a_reward71
def Reward71(popsize, n_ops, window_size, window, gen_window, Off_met, max_gen, decay_reward3, decay_reward4, int_a_reward5, b_reward5, e_reward5, a_reward71, c_reward9, int_b_reward9, int_a_reward9, int_a_reward101, b_reward101, opu, old_opu, total_success, total_unsuccess, old_reward):
    reward = np.zeros(n_ops)
    window = window[window[:, Off_met] != -1][:, :]; #print("inside ", window)
    window_op_sorted, N, rank = count_op(n_ops, window, Off_met)
    for i in range(n_ops):
        if np.any(window[:,0] == i):
            reward[i] += np.sum(window[window[:, 0] == i][:, Off_met]); # print(i, reward[i])
            reward[i] = np.array(reward[i]) / np.array(N[i])
    if np.max(reward) != 0:
        reward = reward / np.max(reward)**a_reward71
    return reward

# index: 9 Applicable for fix number of generations
# Parameter(s): max_gen
def Reward8(popsize, n_ops, window_size, window, gen_window, Off_met, max_gen, decay_reward3, decay_reward4, int_a_reward5, b_reward5, e_reward5, a_reward71, c_reward9, int_b_reward9, int_a_reward9,int_a_reward101, b_reward101, opu, old_opu, total_success, total_unsuccess, old_reward):
    reward = np.zeros(n_ops)
    gen_window = np.array(gen_window)
    if len(gen_window) < max_gen:
        max_gen = len(gen_window)
    for i in range(n_ops):
        for j in range(len(gen_window)-1, len(gen_window)-max_gen-1, -1):
            total_success = 0; total_unsuccess = 0
            #print("first: ",gen_window, np.shape(gen_window), len(gen_window), i, j);print(gen_window[j,0,0])
            if np.any(gen_window[j,:,0] == i):
                total_success, total_unsuccess = count_success(popsize, gen_window, i, j, Off_met)
                if total_success + total_unsuccess != 0:
                    reward[i] += np.sum(gen_window[j, np.where(gen_window[j,:,0] == i) and np.where(gen_window[j, :, Off_met] != -1) , Off_met]) / np.array(total_success + total_unsuccess)
                    #print(reward[i], gen_window[j, np.where(gen_window[j,:,0] == i), Off_met], total_success[j][i])
    # reward = reward / np.array(max_gen)
    return reward



                                                        ##########################Best offspring metric based#############################

# index: 10 Applicable for last two generations
# Parameter(s): c_reward9, int_b_reward9, int_a_reward9
def Reward9(popsize, n_ops, window_size, window, gen_window, Off_met, max_gen, decay_reward3, decay_reward4, int_a_reward5, b_reward5, e_reward5, a_reward71, c_reward9, int_b_reward9, int_a_reward9, int_a_reward101, b_reward101, opu, old_opu, total_success, total_unsuccess, old_reward):
    reward = np.zeros(n_ops);
    best_t = np.zeros(n_ops); best_t_1 = np.zeros(n_ops);
    gen_window = np.array(gen_window);
    
    for i in range(n_ops):
        n_applications = np.zeros(2) # for last 2 generations
        # Calculating best in current generation
        if np.any(gen_window[len(gen_window)-1, :, 0] == i):
            total_success, total_unsuccess = count_success(popsize, gen_window, i, len(gen_window)-1, Off_met)
            n_applications[0] = total_success + total_unsuccess
            if np.any(gen_window[len(gen_window)-1, :, Off_met] != -1):
                best_t[i] = np.max(gen_window[len(gen_window)-1, np.where((gen_window[len(gen_window)-1, :, 0] == i) & (gen_window[len(gen_window)-1, :, Off_met] != -1)), Off_met]); # print(i, best_t[i])
        # Calculating best in last generation
        if len(gen_window)>=2 and np.any(gen_window[len(gen_window)-2,:,0] == i):
            total_success, total_unsuccess = count_success(popsize, gen_window, i, len(gen_window)-2, Off_met)
            n_applications[1] = total_success + total_unsuccess
            if np.any(gen_window[len(gen_window)-2, :, Off_met] != -1):
                best_t_1[i] = np.max(gen_window[len(gen_window)-2, np.where((gen_window[len(gen_window)-2, :, 0] == i) & (gen_window[len(gen_window)-2, :, Off_met] != -1)), Off_met]); # print(i, best_t_1[i])
        if best_t_1[i] != 0 and np.fabs(n_applications[0] - n_applications[1]) != 0:
            reward[i] = c_reward9 * np.fabs(best_t[i] - best_t_1[i]) / (((best_t_1[i])**int_b_reward9) * (np.fabs(n_applications[0] - n_applications[1])**int_a_reward9))
        elif best_t_1[i] != 0 and np.fabs(n_applications[0] - n_applications[1]) == 0:
            reward[i] = c_reward9 * np.fabs(best_t[i] - best_t_1[i]) / ((best_t_1[i])**int_b_reward9)
        elif best_t_1[i] == 0 and np.fabs(n_applications[0] - n_applications[1]) != 0:
            reward[i] = c_reward9 * np.fabs(best_t[i] - best_t_1[i]) / (np.fabs(n_applications[0] - n_applications[1])**int_a_reward9)
        else:
            reward[i] = c_reward9 * np.fabs(best_t[i] - best_t_1[i])
    return reward

# index: 11 Applicable for fix size window
# Parameter(s): window
#def Reward100(popsize, n_ops, window_size, window, gen_window, Off_met, max_gen, decay_reward3, decay_reward4, int_a_reward5, b_reward5, e_reward5, c_reward6, a_reward71, c_reward9, int_b_reward9, int_a_reward9, int_a_reward100, b_reward100, int_a_reward101, b_reward101, opu, old_opu, total_success, total_unsuccess, old_reward):
    #reward = np.zeros(n_ops)
    #for i in range(n_ops):
        #if np.any(window[:,0] == i):
            #reward[i] = np.sum(window[window[:, 0] == i][:, Off_met]); # print(reward[i])
    #return reward

# index: 11 Applicable for fix number of generations
# Parameter(s): max_gen, b_reward101, int_a_reward101
def Reward101(popsize, n_ops, window_size, window, gen_window, Off_met, max_gen, decay_reward3, decay_reward4, int_a_reward5, b_reward5, e_reward5, a_reward71, c_reward9, int_b_reward9, int_a_reward9, int_a_reward101, b_reward101, opu, old_opu, total_success, total_unsuccess, old_reward):
    reward = np.zeros(n_ops); gen_window = np.array(gen_window)
    if len(gen_window) < max_gen:
        max_gen = len(gen_window)
    for i in range(n_ops):
        gen_best = []
        for j in range(len(gen_window)-1, len(gen_window)-max_gen-1, -1):
            # print(gen_window)
            # print("first: ", i, j, np.hstack(gen_window[j, np.where((gen_window[j,:,0] == i) & (gen_window[j, :, Off_met] != -1)), Off_met]))
            if np.any((gen_window[j,:,0] == i) & (gen_window[j, :, Off_met] != -1)):
                # print("inside")
                gen_best.append(np.max(np.hstack(gen_window[j, np.where((gen_window[j,:,0] == i) & (gen_window[j, :, Off_met] != -1)), Off_met])))
                reward[i] += np.sum(gen_best); # print(reward[i])
        if gen_best != []:
            reward[i] = (1/max_gen) * reward[i]**b_reward101 / np.max(gen_best)**int_a_reward101
    return reward


##################################################Quality definitions#####################################################################

# Parameter(s): adaptation_rate
def Quality0(n_ops, adaptation_rate, reward, old_quality, phi, window, scaling_factor, c1_quality6, c2_quality6, discount_rate, delta, Off_met, old_probability, old_reward):
    Q = np.zeros(n_ops); # print(old_quality, reward, old_reward)
    Q = old_quality + adaptation_rate * (reward - old_quality)
    # Q = Q - np.max(Q)
    # Q = np.exp(Q)
    # Q = Q / np.sum(Q)
    return Q

# Parameter(s): scaling_factor
def Quality1(n_ops, adaptation_rate, reward, old_quality, phi, window, scaling_factor, c1_quality6, c2_quality6, discount_rate, delta, Off_met, old_probability, old_reward):
    Q = np.zeros(n_ops)
    window_op_sorted, N, rank = count_op(n_ops, window, Off_met)
    Q = upper_confidence_bound (N, scaling_factor, reward); # print(window_op_sorted, N, rank, reward, Q)
    return Q

# Parameter(s): phi
def Quality2(n_ops, adaptation_rate, reward, old_quality, phi, window, scaling_factor, c1_quality6, c2_quality6, discount_rate, delta, Off_met, old_probability, old_reward):
    Q = np.zeros(n_ops)
    Q = np.exp(reward/phi)
    return Q

def Quality3(n_ops, adaptation_rate, reward, old_quality, phi, window, scaling_factor, c1_quality6, c2_quality6, discount_rate, delta, Off_met, old_probability, old_reward):
    Q = np.zeros(n_ops)
    Q = reward
    return Q

'''def Quality4(n_ops, adaptation_rate, reward, old_quality, phi, window, scaling_factor, discount_rate, delta, Off_met, old_probability, old_reward):
    tran_matrix = normalize_matrix(np.random.rand(n_ops, n_ops)); # print(old_probability)
    tran_matrix = TM(n_ops, old_probability)
    Q = np.matmul(np.linalg.pinv(np.array(1 - discount_rate * tran_matrix)), np.array(reward))
    Q = Q - np.max(Q)
    Q = np.exp(Q)
    # Normalisation for all operators
    Q = Q / np.sum(Q)
    return Q'''

'''def Quality5(n_ops, alpha, reward, old_quality, scaling_factor, window, Off_met, old_probability, old_reward):
    Q = np.zeros(n_ops)
    window_op_sorted, N, rank = count_op(n_ops, window, Off_met)
    Q = (reward + scaling_factor(np.sum(reward)))/N
    return Q'''

# Parameter(s): delta
def Quality4(n_ops, adaptation_rate, reward, old_quality, phi, window, scaling_factor, c1_quality6, c2_quality6, discount_rate, delta, Off_met, old_probability, old_reward):
    Q = np.zeros(n_ops)
    if np.sum(reward) > 0:
        Q = delta * (reward /np.sum(reward))  + (1 - delta) * old_quality
    else:
        Q = delta * n_ops + (1 - delta) * old_quality
    return Q

# Parameter(s): c1_quality6, c2_quality6, discount_rate
def Quality5(n_ops, adaptation_rate, reward, old_quality, phi, window, scaling_factor, c1_quality6, c2_quality6, discount_rate, delta, Off_met, old_probability, old_reward):
    Q = np.zeros(n_ops)
    tran_matrix = TM(n_ops, old_probability)
    tran_matrix = normalize_matrix(np.random.rand(n_ops, n_ops)); # print(old_probability)

    Q = c1_quality6 * reward + c2_quality6 * old_reward
    if discount_rate != 0:
        Q = np.matmul(np.linalg.pinv(np.array(1 - discount_rate * tran_matrix)), np.array(Q))
        Q = Q - np.max(Q)
        Q = np.exp(Q)
        # Normalisation for all operators
        Q = Q / np.sum(Q)
    return Q

##################################################Probability definitions######################################################################

from abc import ABC,abstractmethod

def build_probability(choice, n_ops, prob_args):
    if choice == 0:
        return Probability0(n_ops, prob_args["p_min_prob"], prob_args["e_prob"])
    elif choice == 1:
        return Probability1(n_ops, prob_args["p_min_prob"], prob_args["p_max_prob"], prob_args["beta_prob"])
    elif choice == 2:
        return Probability2(n_ops, prob_args["p_min_prob"], prob_args["beta_prob"])
    elif choice == 3:
        return Probability3(n_ops)
    else:
        raise ValueError("choice {} unknown".format(choice))
 
class ProbabilityType(ABC):
    def __init__(self, n_ops, p_min_prob = None, beta_prob = None):
        self.p_min_prob = p_min_prob
        self.beta_prob = beta_prob
        self.old_probability = np.full(n_ops, 1.0 / n_ops)
        self.eps = np.finfo(self.old_probability.dtype).eps

    def check_probability(self, probability):
        probability /= np.sum(probability)
        assert np.allclose(np.sum(probability), 1.0, equal_nan = True)
        assert np.all(probability >= -0.0)
        # MANUEL: This is wrong! It creates a view of an array and not a copy
        # self.old_probability = self.probability
        self.old_probability[:] = probability
        return(probability)

    @abstractmethod
    def calc_probability(self, quality):
        "Must be implemented by derived probability methods"
        pass
    
# MANUEL: These should have more descriptive names and a doctstring documenting
# where they come from (references) and what they do.
class Probability0(ProbabilityType):
    def __init__(self, n_ops, p_min_prob = 0.1, e_prob = 0.0):
        super().__init__(n_ops, p_min_prob)
        # np.finfo(np.float32).eps adds a small epsilon number that doesn't make any difference but avoids 0.
        self.e_prob = e_prob + self.eps
        # MANUEL: Please do this in every class so one can debug what is actually running.
        debug_print("\n {} : p_min_prob = {}, e_prob = {}".
                    format(type(self).__name__, self.p_min_prob, self.e_prob))
        
    def calc_probability(self, quality):
        probability = self.p_min_prob + (1 - len(quality) * self.p_min_prob) * \
                      (quality + self.e_prob) / np.sum(quality + self.e_prob)
        return super().check_probability(probability)
        

class Probability1(ProbabilityType):
    def __init__(self, n_ops, p_min_prob = 0.1, p_max_prob = 0.9, beta_prob = 0.1):
        super().__init__(n_ops, p_min_prob, beta_prob = beta_prob)
        self.p_max_prob = p_max_prob
        
    def calc_probability(self, quality):
        delta = np.full(quality.shape[0], self.p_min_prob)
        delta[np.argmax(quality)] = self.p_max_prob
        probability = self.old_probability + self.beta_prob  * (delta - self.old_probability)
        probability += self.eps
        return super().check_probability(probability)

class Probability2(ProbabilityType):
    def __init__(self, n_ops, p_min_prob = 0.025, beta_prob = 0.5):
        super().__init__(n_ops, p_min_prob, beta_prob = beta_prob)
        
    def calc_probability(self, quality):
        
        # MANUEL: Why do this?
        if np.sum(quality) != 0:
            quality = quality.copy()
            # Normalize
            quality += self.eps
            quality /= np.sum(quality)

        # np.maximum is element-wise
        probability = self.beta_prob * np.maximum(self.p_min_prob, quality) + (1.0 - self.beta_prob) * self.old_probability[i]
        probability += self.eps
        return super().check_probability(probability)

class Probability3(ProbabilityType):
    def __init__(self, n_ops, ):
        super().__init__(n_ops)
        
    def calc_probability(self, quality):
        probability = quality + self.eps
        return super().check_probability(probability)
    
        
# # Parameters: p_min_prob0, e_prob0
# def Probability0(n_ops, quality, p_min_prob0, e_prob0, p_min_prob1, p_max_prob1, beta_prob1, p_min_prob2, beta_prob2, old_probability):
#     probability = p_min_prob + (1 - len(quality) * p_min_prob0) * ((quality + e_prob0 + np.finfo(np.float32).eps) / (np.sum(quality + e_prob0 + np.finfo(np.float32).eps)))
#     probability = probability/np.sum(probability)
#     return(probability)

# Parameters: beta_prob1, p_min_prob1, p_max_prob1
# def Probability1(n_ops, quality, p_min_prob0, e_prob0, p_min_prob1, p_max_prob1, beta_prob1, p_min_prob2, beta_prob2, old_probability):
#     probability = np.zeros(n_ops)
#     probability = old_probability + beta_prob1 * (p_min_prob1 - old_probability)
#     probability[np.argmax(quality)] = old_probability[np.argmax(quality)] + beta_prob1 * (p_max_prob1 - old_probability[np.argmax(quality)])
#     return ((probability + np.finfo(np.float32).eps) / (np.sum(probability + np.finfo(np.float32).eps)))

# # Parameters: beta_prob2, p_min_prob2
# def Probability2(n_ops, quality, p_min_prob0, e_prob0, p_min_prob1, p_max_prob1, beta_prob1, p_min_prob2, beta_prob2, old_probability):
#     probability = np.zeros(n_ops)
#     #print("qual: ",quality)
#     for i in range(n_ops): # Direct way??????
#         if np.sum(quality) != 0:
#             probability[i] = beta_prob2 * np.max([p_min_prob2, np.array(quality[i] + np.finfo(np.float32).eps) / (np.sum(quality + np.finfo(np.float32).eps))]) + (1 - beta_prob2) * np.array(old_probability[i])
#         else:
#             probability[i] = beta_prob2 * np.max([p_min_prob2, np.array(quality[i])]) + (1 - beta_prob2) * np.array(old_probability[i])
#     #print(probability / np.sum(probability))
#     return ((probability + np.finfo(np.float32).eps) / (np.sum(probability + np.finfo(np.float32).eps)))

# def Probability3(n_ops, quality, p_min_prob0, e_prob0, p_min_prob1, p_max_prob1, beta_prob1, p_min_prob2, beta_prob2, old_probability):
#     probability = np.zeros(n_ops)
#     Probability = quality
#     return ((probability + np.finfo(np.float32).eps) / (np.sum(probability + np.finfo(np.float32).eps)))
    
                              
##################################################Selection definitions######################################################################

def Selection0(op_init_list, p):
    # Roulette wheel selection
    if op_init_list:
        SI = op_init_list.pop()
    else:
        SI = np.random.choice(len(p), p = p)
    return SI


def Selection1(op_init_list, p):
    # Greedy Selection
    if op_init_list:
        SI = op_init_list.pop()
    else:
        SI = np.argmax(p)
    return SI


##################################################Unknown_AOS######################################################################

R_list = [Reward0, Reward1, Reward2, Reward3, Reward4, Reward5, Reward6, Reward70, Reward71, Reward8, Reward9, Reward101]
Q_list = [Quality0, Quality1, Quality2, Quality3, Quality4, Quality5]
#P_list = [Probability0, Probability1, Probability2, Probability3]
S_list = [Selection0, Selection1]

class Unknown_AOS(AOS):
    def __init__(self, popsize, F1, F, u, X, f_min, x_min, best_so_far, best_so_far1, Off_met, Rewar, Qual, Select, n_ops, adaptation_rate, phi, max_gen, scaling_factor, c1_quality6, c2_quality6, discount_rate, delta, decay_reward3, decay_reward4, int_a_reward5, b_reward5, e_reward5, a_reward71, c_reward9, int_b_reward9, int_a_reward9, int_a_reward101, b_reward101, window_size, prob_choice, prob_args):
        # print("Inside unknown AOS-init before super")
        super(Unknown_AOS,self).__init__(popsize, F1, F, u, X, f_min, x_min, best_so_far, best_so_far1, n_ops, window_size)
        # print("Rewar= ",Rewar)
        self.Off_met = Off_met
        # self.Select = Select
        self.adaptation_rate = adaptation_rate
        self.phi = phi
        self.max_gen = max_gen
        self.scaling_factor = scaling_factor
        self.discount_rate = discount_rate
        self.delta = delta
        
        self.decay_reward3 = decay_reward3
        self.decay_reward4 = decay_reward4
        self.int_a_reward5 = int_a_reward5
        self.b_reward5 = b_reward5
        self.e_reward5 = e_reward5
        self.a_reward71 = a_reward71
        self.c_reward9 = c_reward9
        self.int_b_reward9 = int_b_reward9
        self.int_a_reward9 = int_a_reward9
        self.int_a_reward101 = int_a_reward101
        self.b_reward101 = b_reward101
        
        self.c1_quality6 = c1_quality6
        self.c2_quality6 = c2_quality6
        
        self.reward = np.zeros(self.n_ops); self.old_reward = self.reward
        self.quality = np.zeros(self.n_ops); self.quality[:] = 1.0; self.old_quality = self.quality
        self.Reward_fun = R_list[Rewar]
        self.Quality_fun = Q_list[Qual]

        # self.probability = np.zeros(self.n_ops); self.probability[:] = 1.0 / len(self.probability)
        self.probability = np.full(n_ops, 1.0 / n_ops)
        self.probability_type = build_probability(prob_choice, n_ops, prob_args)

        self.Selection_fun = S_list[Select]

        
        
    def Reward(self):
        
        self.reward = self.Reward_fun(self.popsize, self.n_ops, self.window_size, self.window, self.gen_window, self.Off_met, self.max_gen, self.decay_reward3, self.decay_reward4, self.int_a_reward5, self.b_reward5, self.e_reward5, self.a_reward71, self.c_reward9, self.int_b_reward9, self.int_a_reward9, self.int_a_reward101, self.b_reward101, self.opu, self.old_opu, self.total_success, self.total_unsuccess, self.old_reward)
        #print("reward ",self.reward)

    def Quality(self):
        old_probability = self.probability_type.old_probability
        self.quality = self.Quality_fun(self.n_ops, self.adaptation_rate, self.reward, self.old_quality, self.phi, self.window, self.scaling_factor, self.c1_quality6, self.c2_quality6, self.discount_rate, self.delta, self.Off_met, old_probability, self.old_reward); # print("qual: ", self.quality)
        # MANUEL: This is all wrong because they create views, not copies
        self.old_reward = self.reward
        self.old_quality = self.quality
        #print("quality ",self.quality)
        #assert np.sum(self.quality) > 0

    # def Probability(self):
    #     assert len(self.quality) == len(self.probability); # print("p,op1", self.probability, self.old_probability)
    #     self.probability = self.probability_fun(self.n_ops, self.quality, self.p_min_prob0, self.e_prob0, self.p_min_prob1, self.p_max_prob1, self.beta_prob1, self.p_min_prob2, self.beta_prob2, self.old_probability);
    #     #print("current and old probability ", self.probability, self.old_probability)
    #     assert np.allclose(np.sum(self.probability), 1.0, equal_nan = True)
    #     assert np.all(self.probability >= -0.0)
    #     self.old_probability = self.probability

    def Selection(self):
        # print(self.probability, len(self.probability))
        SI = self.Selection_fun(self.op_init_list, self.probability)
        return SI

