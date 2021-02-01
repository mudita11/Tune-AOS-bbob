# Generic-AOS-framework

## Installation

**cocoex, taken from (https://numbbo.github.io/coco-doc/)**

git clone https://github.com/numbbo/coco.git # get coco using git

cd coco

python3 do.py run-python install-user # install Python experimental module cocoex 

python3 do.py install-postprocessing install-user # install post-processing 

**Pygmo**
pip3 install --user pygmo

**Check command**
irace --check

## Description 
**Target runner**
- target-runner-target-vs-fe.py is same as target-runner-hv that calculates the area under the curve generated using trace file. The ECDF graph represents log10(FEvals/dim) vs fraction of targets solved for a problem. 
- target-runner-error-vs-fe.py calculates the area under the curve generated using trace file. The ECDF graph represents log10(FEvals/dim) vs best fitness seen for different targets for a problem. 
- target-runner-best.py receives the best fitness value seen for a problem. 

**default_parameter_setting**
This folder contains default parameter settings of AOS methods from literature.

<!--**exdata folder**
This folder has three subfolders in it: rand_1 (DE1), rand_2 (DE2), rand_best_2 (DE3) and current_rand_1 (DE4). Each of these folders have four folders with data generated by bbob and output. 
- DE*-T-fe: Contains the data for 24 bbob functions when run on DE with strategy represented by * (* can be 1,2,3,4). It has configuration for DE parameters found by target-runner-target-vs-fe.py.  
- DE*-E-fe: Contains the data for 24 bbob functions when run on DE with strategy represented by * (* can be 1,2,3,4). It has configuration for DE parameters found by target-runner-error-vs-fe.py.  
- DE*-B: Contains the data for 24 bbob functions when run on DE with strategy represented by * (* can be 1,2,3,4). It has configuration for DE parameters found by target-runner-best.py. 
- DE*-D: Contains the data for 24 bbob functions when run on DE with strategy represented by * (* can be 1,2,3,4). It has default configuration (FF=0.5, CR = 0.7 and NP = 300) for DE parameters. -->  

## Parameter mapping from code to the document (thesis)
**Reward components and their parameters**
Pareto_Dominance: fix_appl -> fix_appl;
Pareto_Rank: fix_appl -> fix_appl;
Compass_projection: fix_appl -> fix_appl, theta -> mathsymbol(theta);
Area_Under_The_Curve: window_size -> W, decay -> D;
Sum_of_Rank -> window_size -> W, decay -> D;
Success_Rate: max_gen -> max_gen, succ_lin_quad -> mathsymbol(gamma), frac -> Frac, noise -> mathsymbol(epsilon);
Immediate_Success;
Success_sum: max_gen -> max_gen;
Normalised_success_sum_window: window_size -> W, normal_factor -> mathsymbol(omega);
Normalised_success_sum_gen: max_gen -> max_gen;
Best2gen: scaling_constant -> C, alpha -> mathsymbol(alpha), beta -> mathsymbol(beta);
Normalised_best_sum: max_gen -> max_gen, intensity -> mathsymbol(rho), alpha -> mathsymbol(alpha)

**Quality components and their parameters**
Weighted_sum: decay_rate -> mathsymbol(delta);
Upper_confidence_bound: scaling_factor -> c;
Quality_Identity;
Weighted_normalised_sum: decay_rate -> mathsymbol(delta), q_min -> q_min;
Bellman_Equation: weight_reward -> c1, weight_old_reward -> c2, discount_rate -> mathsymbol(gamma)

**Probability components and their parameters**
Probability_Matching: p_min -> p_min, error_prob -> mathsymbol(epsilon);
Adaptive_Pursuit: p_min -> p_min, p_max -> p_max, learning_rate -> mathsymbol(mu);
Probability_Identity

**Selection components and their parameters**
Proportional; 
Greedy; 
Epsilon-Greedy: sel_eps -> eps;
Proportional_Greedy: sel_eps -> eps;
Linear_Annealed
