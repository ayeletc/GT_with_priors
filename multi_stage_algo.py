import os
import time
from unittest import skip
from numpy.random.mtrand import binomial
from tqdm import tqdm
import numpy as np
from datetime import datetime
import itertools
import random
import numpy.matlib
from sample_population import *
from plotters import *
from calc_bounds_and_num_of_tests import *
from Markov_model import *
import scipy.io


#%% Config simulation
N                   = 100 # for Markov: N=500, K=3 ; N=1024, K=8; N=10000, K=13 
vecK                = [3]#[6,8,10,12,14,16,18,20,22,24]
nmc                 = 100000
'''
For n=500,K=3:
if the criteria is whether the defective set is in the estimatons of lva or no, then
[0.5, 0.6,0.7] can reach the limit (with L=50 trajectories), and 0.8 also can reach 
it but it takes more time.
if the criteria is whether the gt including lva+map reached a successful detection of 
the defective set, then we know that we reach very high success rates so only 
for ~0.5 and 0.6 we can collect enough data in a reasonable time
When L=100, we can't collect data even for 0.7 in a reasonable time
'''
enlarge_tests_num_by_factors = [0.6]#[0.5, 0.6, 0.7]#[0.5, 0.6, 0.7,0.8,0.9,1.0]#[0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.1]#[0.5, 0.75, 1, 1.25] #[0.5, 0.25, 0.5, 0.75, 1, 1.5]#[0.75, 0.8, 0.9, 1, 1.25, 1.5, 1.75, 2]# [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.3, 1.7, 2] #[0.85, 0.9, 0.95, 1, 1.25, 1.5, 1.75, 2]#
Tbaseline           = 'ML' # options: 'ML', 'lb_no_priors', 'lb_with_priors', 'GE'
methods_DD          = ['Normal']#{'Normal', 'Sum'} # options: Normal, Iterative, Sum
third_step_type     = 'viterbi+MAP' # options: ['MAP', 'MLE', 'MAP_for_GE_all_options', 'MAP_for_GE_stop_search', 'MAP_for_GE_use_sortedPw', 'viterbi', 'viterbi+MAP']
permutation_factor  = 1 # compared [10, 20 , 50, 100] for N=100, K=2,4,6. 50 was the most effective
save_raw            = True
save_fig            = False
save_path           = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
is_plot             = True
do_third_step       = True
is_sort_comb_by_priors = True
add_dd_based_prior  = False
debug_mode          = False 
plot_status_DD      = False

### probabilistic model config ###
sample_method       = 'GE'  # options: 'GE', 'Markov'

### code config ###
code_type           = 'bernoulli' # options: 'bernoulli', 'typical_bernoulli', 'near_constant'
delta_typical_cols  = 0.1# for N=100,K=3 and T=0.75ML 
delta_typical_rows  = 0.1

### viterbi config ###
extend_obsesrvations = True # with 1 step there is not much difference between true/false here, if true fix the paths(cut the last item in the traj_paths)
do_map_if_viterbi_fail = False
init_paths_number    = 50 # initial number of paths to find in list viterbi algorithm (in the 1st iiteration)
max_paths_for_lva    = 50 # TODO: try 40 and 60 (already done 80)
step_in_lva_paths    = 50
viterbi_time_steps   = 1 #1/2/3
viterbi_comb_threshold = [4*3,6,6,6,6,6]#30 # 30 = K*10=> gamma = 10 = N/50 = N^0.03705 = 0.02*N; 
max_iteration_for_map = 1e6
gamma_dict = {'0.5': 12,
              '0.6': 6,
              '0.7': 6,
              '0.8': 6,
              '0.9': 6,
              '1.0': 6,
              }
### seed ###
# random.seed(123)
# np.random.seed(123)
random.seed(456)
np.random.seed(456)

invalid = -1
all_permutations = []
vecTs = []
print('===== Sample method: {} ====='.format(sample_method))
for method_DD in methods_DD:
    print('===== DD method: {} || code = {} ====='.format(method_DD, code_type))
    if do_third_step:
        print('===== Last step decoder: {} ====='.format(third_step_type))
        if third_step_type == 'viterbi+MAP':
            print('===== max paths in lva = {}  <=>  max lva iterations = {} ====='.format(max_paths_for_lva, int((max_paths_for_lva-init_paths_number)/step_in_lva_paths)+1))
            print('===== viterbi time steps = {} ====='.format(viterbi_time_steps))
            print('===== gammaK = {} ====='.format(viterbi_comb_threshold))
    else:
        print('===== Stop after DD ====')
    if save_fig:
        print('==== save_fig TRUE ====')
    else:
        print('==== save_fig FALSE ====')
    ## Initialize counters
    numOfK = len(vecK)
    viterbi_fail_try_full_map = False
    num_of_test_scale = len(enlarge_tests_num_by_factors)
    count_DND1 = np.zeros((numOfK, num_of_test_scale, nmc))
    count_PD1 = np.zeros((numOfK, num_of_test_scale, nmc))
    count_DD2 = np.zeros((numOfK, num_of_test_scale, nmc))
    count_unknown2 = np.zeros((numOfK, num_of_test_scale, nmc))
    count_success_DD_exact = np.zeros((numOfK, num_of_test_scale, nmc))
    count_success_DD_non_exact = np.zeros((numOfK, num_of_test_scale, nmc))
    count_success_exact_third_step = np.zeros((numOfK, num_of_test_scale, nmc))
    count_success_non_exact_third_step = np.zeros((numOfK, num_of_test_scale, nmc))
    count_not_detected = np.zeros((numOfK, num_of_test_scale))
    expected_notDetected = np.zeros((numOfK, num_of_test_scale))
    expected_DD = np.zeros((numOfK, num_of_test_scale))
    expected_PD = np.zeros((numOfK, num_of_test_scale))
    expected_unknown = np.zeros((numOfK, num_of_test_scale))
    if debug_mode:
        bound_PD_DND = np.zeros((numOfK, num_of_test_scale))
        bound_DD_DD = np.zeros((numOfK, num_of_test_scale))
    count_num_of_paths_in_viterbi = -1*np.ones((numOfK, num_of_test_scale, nmc))
    count_num_of_paths_in_viterbi_multi_steps = -1*np.ones((numOfK, num_of_test_scale, nmc))
    count_viterbi_found_zero_options = np.zeros((numOfK, num_of_test_scale))
    count_viterbi_fail_try_full_map = np.zeros((numOfK, num_of_test_scale))
    if viterbi_fail_try_full_map:
        count_viterbi_fail_and_full_map_fail = np.zeros((numOfK, num_of_test_scale))
    count_viterbi_fail_cant_try_map = np.zeros((numOfK, num_of_test_scale))
    num_of_comb_in_map = np.zeros((numOfK, num_of_test_scale, nmc))
    count_viterbi_relevant_paths = np.zeros((numOfK, num_of_test_scale, nmc))
    Pw_of_true_out_of_max_Pw = np.zeros((numOfK, num_of_test_scale, nmc))
    paths_with_less_than_K = np.zeros((numOfK, num_of_test_scale, nmc))
    
    # Dataset for ViterbiNet
    n_samples = int(nmc/10)
    half_n_samples = int(n_samples/2)
    # dataset_observations_suc = np.zeros((numOfK, num_of_test_scale, 1, N, half_n_samples)).astype(np.uint8)
    # dataset_estimations_suc = np.zeros((numOfK, num_of_test_scale, step_in_lva_paths, N, half_n_samples)).astype(np.uint8)
    # dataset_observations_fail = np.zeros((numOfK, num_of_test_scale, 1, N, half_n_samples)).astype(np.uint8)
    # dataset_estimations_fail = np.zeros((numOfK, num_of_test_scale, step_in_lva_paths, N, half_n_samples)).astype(np.uint8)
    
    # dataset_partial_recovery_suc = np.zeros((numOfK, num_of_test_scale, half_n_samples)).astype(np.float32) # detection rate
    # dataset_partial_recovery_fail = np.zeros((numOfK, num_of_test_scale, half_n_samples)).astype(np.float32) # detection rate

    dataset_observations = np.zeros((numOfK, num_of_test_scale, 1, N, n_samples)).astype(np.uint8)
    dataset_estimations = np.zeros((numOfK, num_of_test_scale, step_in_lva_paths, N, n_samples)).astype(np.uint8)
    dataset_is_set_in_estimations = np.zeros((numOfK, num_of_test_scale, n_samples)).astype(np.uint8)
    dataset_is_detected = np.zeros((numOfK, num_of_test_scale, n_samples)).astype(np.uint8)
    dataset_true_defective_set = np.zeros((numOfK, num_of_test_scale, n_samples, 1, N)).astype(np.uint8)
    # dataset_partial_recovery = np.zeros((numOfK, num_of_test_scale, n_samples)).astype(np.float32) # detection rate
    
    #%% Start simulation
    for idxK in range(numOfK):
        K = vecK[idxK]
        print('K = ' + str(K))
        # For each K calculate number of test according the Tml and scale factor
        ge_model = None
        markov_model = None
        if sample_method == 'GE': # create the ge model 
            _, ge_model = sample_population_gilbert_elliot_channel(N, K, ge_model, debug=False)
        elif sample_method == 'Markov':
            if N == 500:
                _, markov_model = sample_population_for_N500_K3_ts3(N, K, markov_model)
            elif N == 1024:
                _, markov_model = sample_population_for_N1024_K8_ts3(N, K, markov_model)
            elif N == 10000:
                _, markov_model = sample_population_for_N10000_K13_ts3(N, K, markov_model)
            else:
                print('Sampling method is not defined for the given N,K')
        vecT = calculate_vecT_for_K(K, N, enlarge_tests_num_by_factors, Tbaseline=Tbaseline, Pe=Pe, 
                    sample_method=sample_method, ge_model=ge_model, Pu=None, coeff_mat=None)
        vecTs.append(vecT)
        time_start = time.time()
        for idxT in range(num_of_test_scale):
            T = np.int16(vecT[idxT])
            print('T = {} || factor = {}'.format(T, enlarge_tests_num_by_factors[idxT]))
            idx_viterbinet_suc = 0
            idx_viterbinet_fail = 0
            idx_viterbinet_in_estimations = 0
            idx_viterbinet_not_in_estimations = 0
            idx_viterbinet = 0
            overflow_const = 1 #10^T

            p = 1-2**(-1/K) # options: 1/K, log(2)/K, 1-2**(-1/K)
            expected_PD[idxK, idxT] = K + (N-K) * (1-p*(1-p)**K)**T 
            nPD = expected_PD[idxK, idxT]
            expected_DD[idxK, idxT]  = K*(1-(1-p*(1-p)**(nPD-1))**T)#nPD*p_defective*(1-(1-p*(1-p)**(nPD-1))**T) # version1 - p_defective appears once
            expected_notDetected[idxK, idxT] = K - expected_DD[idxK, idxT]

            alpha = 1.4 * enlarge_tests_num_by_factors[idxT]
            if debug_mode:
                fn = N ** (-0.5*alpha*(1-np.log(2)/K))
                bound_PD_DND[idxK, idxT] = K + (N-K) * fn
                gn = N ** (-0.5*alpha*(1-np.log(2)/K) ** (fn * N))
                bound_DD_DD[idxK, idxT] = K * (1-gn)

            count_success_DD_exact_vec_nmc = np.zeros((nmc,))
            count_success_DD_non_exact_vec_nmc = np.zeros((nmc,))
            if sample_method == 'GE':
                ge_model.calculate_num_of_permutations_by_entropy(K, T, nPD)
            if code_type == 'near_constant':
                L = np.int16(np.round(p*T))
                rows_idx = np.arange(T)
            
            for nn in tqdm(range(nmc), desc='T ' + str(T)):
            # for nn in range(nmc):
                if debug_mode:
                    print('nn', nn)
                ## Sample
                if sample_method == 'onlyPu':
                    U, Pu, Pw = sample_population_no_corr(N, K)
                elif sample_method == 'indicative':
                    U, Pu, Pw, num_of_distractions = sample_population_indicative(N, K)
                elif sample_method == 'GE':
                    U, ge_model = sample_population_gilbert_elliot_channel(N, K, ge_model, debug=False)
                elif sample_method == 'Markov':
                    U, markov_model = sample_population_for_N500_K3_ts3(N, K, markov_model)
                true_defective_set = np.where(U == 1)[1].tolist()
                ## 1. Definitely Not Defective
                # Encoder - bernoulli 
                if 'bernoulli' in code_type:
                    X =  np.multiply(np.random.uniform(0,1,(T, N)) < p,1) # iid testing matrix
                
                if code_type == 'typical_bernoulli':
                    all_X_typical = False
                    while not all_X_typical:
                        non_typical_cols = set(np.arange(N))
                        while non_typical_cols:
                            non_typical_cols_list = list(non_typical_cols)
                            X[:,non_typical_cols_list] = np.multiply(np.random.uniform(0,1,(T, len(non_typical_cols))) < p,1)
                            # if x.shape[1] == 1 => np.sum without axis/addaxis
                            sum_cols = np.sum(X[:,non_typical_cols_list], axis=0)
                            dist_cols = np.abs(sum_cols / T - p)
                            typical_cols = np.array(non_typical_cols_list)[np.where(dist_cols < delta_typical_cols)[0]]
                            non_typical_cols = non_typical_cols.difference(set(typical_cols))
                            
                        # verify that the rows are also typical 
                        sum_rows = np.sum(X, axis=1)
                        dist_rows = np.abs(sum_rows / N - p)
                        non_typical_row = [r for r in dist_rows if r > delta_typical_rows]
                        all_X_typical = len(non_typical_row) == 0

                    non_typical_cols = list(non_typical_cols)

                elif code_type == 'near_constant':
                    X = np.zeros((T, N)).astype(np.uint8)
                    for ii in range(N):
                        X[random.choices(population=rows_idx, k=L),ii] = 1

                tested_mat = X*U
                Y = np.sum(tested_mat, 1) > 0 

                X_mark_occlusion = np.zeros((X.shape[0]+1, X.shape[1]))
                X_mark_occlusion[1:, :] = X
                X_mark_occlusion[0,true_defective_set] = 5 
                
                # Decoder - CoMa
                PD1 = np.arange(N)
                DND1 = []
                for ii in range(T):
                    if len(PD1) <= K:
                        break 
                    if Y[ii] == 0:
                        for jj in PD1:
                            # iter_until_detection_CoMa_and_DD[idxK, idxT] += 1
                            if X[ii,jj] == 1: # definitely not defected
                                PD1 = PD1[PD1 != jj]
                                DND1 += [jj]
                count_DND1[idxK, idxT, nn] = len(DND1)
                count_PD1[idxK, idxT, nn] = len(PD1)
                
                if len(PD1) <= K: # all the PD are DD - all defective found
                    # TODO: yul I think its ok but maybe I should check it in case the code gets here
                    count_DD2[idxK, idxT, nn] = len(PD1)
                    count_success_DD_exact[idxK, idxT, nn] += 1
                    count_success_DD_non_exact[idxK, idxT, nn] += 1
                    continue
                ## 2. Definite Defective
                # steps 1&2
                if method_DD == 'Normal':
                    DD2 = []
                    for ii in range(T):
                        if Y[ii] == 1 and np.sum(X[ii,PD1]) == 1: # only 1 item among the PD equals 1 and the rest equal 0
                            jj = np.where(X[ii,PD1] == 1)[0][0] # find the definite defective item index in PD1 array
                            # iter_until_detection_CoMa_and_DD[idxK, idxT] += len(PD1)

                            defective = PD1[jj]
                            if defective not in DD2: # add jj only if jj is not already detected as DD
                                DD2 += [defective]
                                X_mark_occlusion[ii+1, defective] = 4
                        elif Y[ii]==1: # occlusion - mark
                            participating = PD1[np.where(X[ii,PD1] ==1)[0]]
                            X_mark_occlusion[ii+1, participating] = 2
                            defective_occluded = [e for e in participating if e in true_defective_set]
                            X_mark_occlusion[ii+1, defective_occluded] = 3
                    
                    count_DD2[idxK, idxT, nn] = len(DD2)

                    if len(DD2) >= K: # all defective found
                        count_success_DD_exact[idxK, idxT, nn] += 1
                        count_success_DD_non_exact[idxK, idxT, nn] +=1
                        continue
                    
                count_not_detected_defectives = K-len(DD2)
                count_success_DD_non_exact[idxK, idxT, nn] += (len(DD2) / K)
                unknown2 = [e for e in PD1 if e not in DD2]#PD1[PD1 not in DD2][0]

                count_unknown2[idxK, idxT, nn] = len(unknown2)

                if debug_mode:
                    print('#PD = {} || #DD = {} || #unknown = {}'.format(len(PD1), len(DD2), len(unknown2)))        
                
                ## 3.
                if is_plot and plot_status_DD:
                    plot_status_before_third_step(N, K, T, enlarge_tests_num_by_factors[idxT], PD1, DD2, true_defective_set) 
                # hmm_model = ge_model.model_as_hmm(K, T, len(PD1), p)
                # hmm_model_2steps = ge_model.model_as_hmm_with_2_steps_memory(K,T,len(PD1), p)
                observations = 2*np.ones((N,)).astype(np.int8)
                observations[DD2] = 1
                observations[DND1] = 0
                # observations[PD1] = 2

                if not do_third_step:
                    continue
                estU = np.zeros(U.shape)
                if third_step_type == 'viterbi':
                    map_trajectory, map_probabilities = hmm_model.viterbi_algo_adjusted_to_GE(observations)
                    
                    map_trajectory = np.array(map_trajectory)
                    estU[0,map_trajectory == 1] = 1
                    # senity check - is DD2 in the most likely path?
                    if len(DD2) > 0 and len(list(set(DD2) - set(np.where(map_trajectory == 1)[0]))) != 0:
                        print('{} DD2 not in path'.format(nn)) 

                elif third_step_type == 'viterbi+MAP':
                    possible_combination_found = False
                    skip_viterbi_paths_options = False
                    if debug_mode:
                        print('start lva')  
                    top_k = init_paths_number
                    while not possible_combination_found and top_k <= max_paths_for_lva:
                        if viterbi_time_steps == 1:
                            if sample_method == 'GE':
                                hmm_model = ge_model.model_as_hmm(K, T, len(PD1), p, ver_states=False)
                                path_trajs, path_probs, ml_prob, ml_traj = hmm_model.list_viterbi_algo_parallel_with_deter(observations, top_k=top_k)
                            elif sample_method == 'Markov':
                                # hmm_model_1step_memory = markov_model.model_as_hmm_with_1step_memory()
                                hmm_model = markov_model.model_as_hmm_with_1step_memory()
                                path_trajs, path_probs, ml_prob, ml_traj = hmm_model.list_viterbi_algo_parallel_with_deter(observations, top_k=top_k)
                            else: 
                                print('Not defined yet')

                            # remove dup rows
                            unique_rows = np.unique(path_trajs, axis=0)
                            
                            paths = unique_rows
                            count_num_of_paths_in_viterbi[idxK, idxT, nn] = paths.shape[0]
                            if debug_mode:
                                print(paths.shape[0], ' unique paths')
                        else:
                            if sample_method == 'GE':
                                hmm_model = ge_model.model_as_hmm_with_long_memory(K,T,len(PD1), p, ts=viterbi_time_steps)
                            elif sample_method == 'Markov':
                                hmm_model = markov_model.model_as_hmm()
                            
                            path_trajs2, path_probs2, ml_prob2, ml_traj2 = hmm_model.list_viterbi_algo_parallel_with_long_memory(observations, top_k=top_k)
                            paths = np.copy(path_trajs2)

                            # remove dup rows
                            unique_rows = np.unique(paths, axis=0)
                            if debug_mode and unique_rows.shape[0] != paths.shape[0]:
                                print('check')
                            paths = unique_rows
                            count_num_of_paths_in_viterbi_multi_steps[idxK, idxT, nn] = paths.shape[0]
                        
                        count_detections_for_non_exact_recovery = []
                        
                        # prepare k-defective optional sets
                        optional_sets_list = []
                        for ii in range(paths.shape[0]):
                            # if (len(optional_sets_list) > math.comb(len(unknowns), K))
                                # break
                            detected_defective_set = np.where(paths[ii]==1)[0]
                            # viterbi_comb_threshold = max(expected_PD[idxK, idxT], viterbi_comb_threshold)
                            if detected_defective_set.shape[0] >= gamma_dict[str(enlarge_tests_num_by_factors[idxT])]:#N*0.9:
                                # Too many potential combinations, skip the viterbi option
                                skip_viterbi_paths_options = True
                                # TODO: maybe already here we need the itertools&MAP?
                                continue
                            
                            elif detected_defective_set.shape[0] >= K:
                                count_viterbi_relevant_paths[idxK, idxT, nn] += 1
                                # reasonable number of combinations, find the options
                                detected_defective_set_minus_DD = [e for e in detected_defective_set if e not in DD2] 
                                possible_kleft_combinations = prepare_nchoosek_comb(detected_defective_set_minus_DD, K-len(DD2))
                                possible_k_combinations = []
                                for c in possible_kleft_combinations:
                                    c = (list(c) + DD2)
                                    possible_k_combinations.append(c)
                                
                                # possible_k_combinations = prepare_nchoosek_comb(detected_defective_set.tolist(), K)
                                for comb in possible_k_combinations:
                                    list_of_false_positive_items = [item for item in detected_defective_set if (item not in unknown2) and (item not in DD2)]
                                    if (not DD2 or len(list(set(DD2) - set(comb))) == 0) and not list_of_false_positive_items: # the comc include all the DD2 and does not include dnd1
                                        optional_sets_list.append(list(comb))
                            else: # no path with as least K defectives:
                                paths_with_less_than_K[idxK,idxT,nn] += 1/paths.shape[0]

                                if len(count_detections_for_non_exact_recovery) <= len(detected_defective_set):
                                    count_detections_for_non_exact_recovery = list(detected_defective_set)
                        
                        if optional_sets_list: 
                            possible_combination_found = True
                        else:
                            top_k += step_in_lva_paths
                    
                    if possible_combination_found:
                        # keep only unique combinations, remove dups
                        optional_sets_ar = np.array(optional_sets_list)
                        optional_sets_ar = np.unique(optional_sets_ar, axis=0).astype(np.uint16)#.tolist()
                        viterbi_fail_try_full_map = False 
                        # num_comb_after_lva[idxK, idxT,nn]
                    
                    elif top_k > max_paths_for_lva:
                        # didn't find valid options using viterbi
                        # check if going over all the options is possible:
                        count_viterbi_found_zero_options[idxK, idxT] += 1 
                        ''' Record for viterbinet dataset'''
                        if idx_viterbinet_not_in_estimations < half_n_samples:
                            dataset_observations[idxK, idxT, 0, :, idx_viterbinet] = observations
                            dataset_estimations[idxK, idxT, :, :, idx_viterbinet] = path_trajs2
                            dataset_is_set_in_estimations[idxK, idxT, idx_viterbinet] = 0
                            dataset_is_detected[idxK, idxT, idx_viterbinet] = 0
                            dataset_true_defective_set[idxK, idxT, idx_viterbinet, :] = U
                            idx_viterbinet += 1 
                            idx_viterbinet_not_in_estimations +=1 

                        num_of_true_set_options_in_step3 = int(scipy.special.comb(len(unknown2), K-len(DD2)))
                        viterbi_fail_try_full_map = False # initialization
                        if do_map_if_viterbi_fail and num_of_true_set_options_in_step3 <= max_iteration_for_map:
                            optional_sets_ar = np.fromiter(itertools.chain(*itertools.combinations(unknown2, count_not_detected_defectives)), np.uint16).reshape((-1,count_not_detected_defectives))
                            count_viterbi_fail_try_full_map[idxK, idxT] += 1 
                            viterbi_fail_try_full_map = True
                        else:
                            count_viterbi_fail_cant_try_map[idxK, idxT] += 1  # too many options
                            estU = np.zeros(U.shape)
                            estU[0,count_detections_for_non_exact_recovery] = 1
                            detected_defectives = np.where(estU==1)[1] # may be errornous detection
                            not_detected = set(true_defective_set)-set(detected_defectives)
                            num_of_correct_detection = K-len(not_detected)
                            count_success_non_exact_third_step[idxK, idxT, nn] += (num_of_correct_detection-len(DD2))/K 
                            continue
                    else:
                        pass

                    if debug_mode:
                        print('start map, #options=', optional_sets_ar.shape[0])
                    # MAP
                    ## 1st option - iterative MAP:
                    num_of_comb_in_map[idxK, idxT, nn] = optional_sets_ar.shape[0]
                    apriori = invalid*np.ones((optional_sets_ar.shape[0],))
                    for comb_idx, comb in enumerate(optional_sets_ar):
                        comb = comb.tolist()
                        U_forW = np.zeros((1,N))
                        U_forW[0,list(set(comb + DD2))] = 1
                        
                        X_forW = X*U_forW
                        Y_forW = np.sum(X_forW, 1) > 0
                        if (Y_forW != Y).any():
                            if debug_mode and set(comb+DD2) == set(true_defective_set):
                                print('Yw!=Y')
                            continue
                        if sample_method == 'GE':
                            Pw_map = ge_model.calc_Pw_fixed(N, comb, DD2, DND1)
                        elif sample_method == 'Markov':
                            Pw_map = markov_model.calc_Pw(N, comb, DD2, DND1)
                        # Pw_vietrbi = TODO:get the path probability and compare to the map's
                        P_X_Sw = p ** np.sum(X_forW == 1)
                        apriori[comb_idx] = Pw_map# * P_X_Sw
                    
                    max_likelihood_W = np.argmax(apriori)
                    estU = np.zeros(U.shape)
                    estU[0,optional_sets_ar[max_likelihood_W,:].tolist() + DD2] = 1
                    
                elif third_step_type == 'MAP_for_GE_all_options':
                    all_permutations = np.array(list(itertools.combinations(unknown2, count_not_detected_defectives)))
                    num_of_permutations = all_permutations.shape[0]
                    apriori = invalid*np.ones((num_of_permutations,1)) 
                    for comb in range(num_of_permutations):                            
                        permute = all_permutations[comb,:].tolist()
                        U_forW = np.zeros((1,N))
                        U_forW[0,permute + DD2] = 1
                        X_forW = X*U_forW
                        Y_forW = np.sum(X_forW, 1) > 0

                        if (Y_forW != Y).any():
                            if debug_mode and set(permute+DD2) == set(true_defective_set):
                                print('Yw!=Y')
                            continue
                        
                        Pw = ge_model.calc_Pw_fixed(N, permute, DD2, DND1)
                        P_X_Sw = p ** np.sum(X_forW == 1)
                        apriori[comb] = Pw * P_X_Sw
                        if set(permute+DD2) == set(true_defective_set) and debug_mode:
                            print('true defective set prior: prior(W*) = ' + str(apriori[comb,0]))
                    
                    max_likelihood_W = np.argmax(apriori)
                    if debug_mode:
                        print('chosen defective set prior: prior(estW) = ' + str(apriori[max_likelihood_W,0]))
                    estU = np.zeros(U.shape)
                    estU[0, all_permutations[max_likelihood_W,:]] = 1  
                    estU[0, DD2] = 1

                elif third_step_type == 'MAP_for_GE_use_sortedPw':
                    all_permutations3, Pw_sorted, num_of_iterations_in_sort = ge_model.sort_comb_by_priors_GE_cut_by_entropy(N, K, T, nPD, DD2, DND1, unknown2, permutation_factor=permutation_factor)
                    num_of_permutations3 = all_permutations3.shape[0]
                    num_of_comb_in_map[idxK, idxT,nn] = num_of_permutations3
                    apriori = invalid*np.ones((num_of_permutations3,1))
                    already_had_a_match = False
                    idx_first_match = -1
                    
                    for comb in range(Pw_sorted.shape[0]):                            
                        permute = all_permutations3[comb,:].tolist()
                        U_forW = np.zeros((1,N))
                        U_forW[0,permute + DD2] = 1
                        X_forW = X*U_forW
                        Y_forW = np.sum(X_forW, 1) > 0

                        if (Y_forW != Y).any():
                            if debug_mode and set(permute+DD2) == set(true_defective_set):
                                print('Yw!=Y')
                            continue
                        P_X_Sw = p ** np.sum(X_forW == 1)
                        apriori[comb] = Pw_sorted[comb] * P_X_Sw
                        
                        if set(permute+DD2) == set(true_defective_set) and debug_mode:
                            print('true defective set prior: prior(W*) = ' + str(apriori[comb,0]))
                    # iter_until_detection_third_step_full[idxK, idxT] += Pw_sorted.shape[0]
                    max_likelihood_W = np.argmax(apriori)
                    if idx_first_match != max_likelihood_W:
                        print('we took another comb!')
                    if debug_mode:
                        print('chosen defective set prior: prior(estW) = ' + str(apriori[max_likelihood_W,0]))
                    estU = np.zeros(U.shape)
                    estU[0, all_permutations3[max_likelihood_W,:]] = 1  
                    estU[0, DD2] = 1
                    # iter_until_detection_third_step_eff[idxK, idxT] += max_likelihood_W
                    Pw_of_true_out_of_max_Pw[idxK, idxT, nn] = Pw_sorted[max_likelihood_W] / Pw_sorted[0]

                elif third_step_type == 'MAP_for_GE_stop_search':
                    all_permutations3, Pw_sorted, num_of_iterations_in_sort = ge_model.sort_comb_by_priors_GE_cut_by_entropy(N, K, T, nPD, DD2, DND1, unknown2, permutation_factor=permutation_factor)
                    # iter_until_detection_third_step_full[idxK, idxT] += num_of_iterations_in_sort
                    # iter_until_detection_third_step_eff[idxK, idxT] += num_of_iterations_in_sort
                    num_of_permutations3 = all_permutations3.shape[0]
                    apriori = invalid*np.ones((Pw_sorted.shape[0],1))
                    idx_of_match = -1
                    # print('num_of_permutations3', num_of_permutations3)
                    for comb in range(Pw_sorted.shape[0]):                            
                        permute = all_permutations3[comb,:].tolist()
                        U_forW = np.zeros((1,N))
                        U_forW[0,permute + DD2] = 1
                        X_forW = X*U_forW
                        Y_forW = np.sum(X_forW, 1) > 0

                        if not (Y_forW != Y).any(): # Yw match Y
                            # stop searching - take the one with the highest prior that match Y
                            estU = U_forW
                            max_likelihood_W = comb
                            Pw_of_true_out_of_max_Pw[idxK, idxT, nn] = Pw_sorted[comb] / Pw_sorted[0]
                    if estU is None: # not supposed to get here
                        # dummies
                        max_likelihood_W = 0
                        estU = np.zeros(all_permutations3[0,:])
                        Pw_of_true_out_of_max_Pw[idxK, idxT, nn] = 0

                elif third_step_type == 'MLE':
                    try:
                        all_permutations3_no_prior = np.array(list(itertools.combinations(unknown2, count_not_detected_defectives)))
                        num_of_all_permutations3_no_prior = all_permutations3_no_prior.shape[0]
                    except:
                        print('could not find permutations')
                        continue
                    # we want to find estW: estW = argmax{P(Y|W)}
                    max_likelihood_W = None
                    min_error_counter = np.inf
                    # try each permutation w: 
                    for w, permute in enumerate(all_permutations3_no_prior):
                        # calculate Y for this permutation Y|W=w
                        U_forW = np.zeros((1,N))
                        U_forW[0,permute.tolist()+DD2] = 1
                        X_forW = X*U_forW
                        Y_forW = np.sum(X_forW, 1) > 0
                        # evaluate Y
                        error_counter = np.sum(Y_forW != Y)
                        if error_counter < min_error_counter:
                            min_error_counter = error_counter
                            max_likelihood_W = w
                            if error_counter == 0: # Yw = Y
                                break 
                    estU = np.zeros(U.shape)
                    estU[0, all_permutations3_no_prior[max_likelihood_W,:]] = 1  
                    estU[0, DD2] = 1
                    
                #%% Calc success rate
                if np.sum(U != estU) == 0:
                    count_success_exact_third_step[idxK, idxT, nn] += 1
                    count_success_non_exact_third_step[idxK, idxT, nn] += (K-len(DD2))/K
                    if viterbi_fail_try_full_map:
                        print('here')
                else:
                    if viterbi_fail_try_full_map:  # full map also failed in terms of exact analysis
                        count_viterbi_fail_and_full_map_fail[idxK, idxT] += 1


                    # count only the items detected in the 3rd step 
                    detected_defectives = np.where(estU==1)[1] # may be errornous detection
                    not_detected = set(true_defective_set)-set(detected_defectives)
                    num_of_correct_detection = K-len(not_detected)
                    count_success_non_exact_third_step[idxK, idxT, nn] += (num_of_correct_detection-len(DD2))/K 
                    # count_not_detected_map_error[idxK, idxT, nn] += 1
                    # true defective set was in optional_sets_list?
                    for opt_set in optional_sets_list:
                        if set(true_defective_set) - set(opt_set) == set([]):
                            pass
            
                # Dateset for ViterbiNet
                if idx_viterbinet_in_estimations < half_n_samples or idx_viterbinet_not_in_estimations < half_n_samples:
                    # check if true defective set is in estimations:
                    true_defective_set_in_estimations = False
                    for r in range(path_trajs2.shape[0]):
                        path = path_trajs2[r,:]
                        estimated_defectives = set(np.where(path==1)[0])
                        if set(true_defective_set).intersection(set(estimated_defectives)) != set():
                            true_defective_set_in_estimations = True
                            if idx_viterbinet_in_estimations < half_n_samples:
                                dataset_observations[idxK, idxT, 0, :, idx_viterbinet] = observations
                                dataset_estimations[idxK, idxT, :, :, idx_viterbinet] = path_trajs2
                                dataset_is_set_in_estimations[idxK, idxT, idx_viterbinet] = 1#int(true_defective_set_in_estimations)
                                dataset_is_detected[idxK, idxT, idx_viterbinet] = count_success_exact_third_step[idxK, idxT, nn]
                                dataset_true_defective_set[idxK, idxT, idx_viterbinet,:] = U
                                idx_viterbinet_in_estimations += 1
                                idx_viterbinet += 1
                            break
                    if not true_defective_set_in_estimations and idx_viterbinet_not_in_estimations < half_n_samples:
                        dataset_observations[idxK, idxT, 0, :, idx_viterbinet] = observations
                        dataset_estimations[idxK, idxT, :, :, idx_viterbinet] = path_trajs2
                        dataset_is_set_in_estimations[idxK, idxT, idx_viterbinet] = int(true_defective_set_in_estimations)
                        dataset_is_detected[idxK, idxT, idx_viterbinet] = count_success_exact_third_step[idxK, idxT, nn]
                        dataset_true_defective_set[idxK, idxT, idx_viterbinet,:] = U
                        idx_viterbinet_not_in_estimations += 1
                        idx_viterbinet += 1
                        # dataset_observations[idxK, idxT, 0, :, idx_viterbinet] = observations
                        # dataset_estimations[idxK, idxT, :, :, idx_viterbinet] = path_trajs2
                        # dataset_is_set_in_estimations[idxK, idxT, idx_viterbinet] = int(true_defective_set_in_estimations)
                        # dataset_is_detected[idxK, idxT, idx_viterbinet] = count_success_exact_third_step[idxK, idxT, nn]

                '''
                if count_success_exact_third_step[idxK, idxT, nn] == 1:
                    # success <=> the true_defective_set in the estimations
                    if idx_viterbinet_suc < half_n_samples:
                        dataset_observations_suc[idxK, idxT, 0, :, idx_viterbinet_suc] = observations
                        dataset_estimations_suc[idxK, idxT, :, :, idx_viterbinet_suc] = path_trajs2
                        dataset_partial_recovery_suc[idxK, idxT, idx_viterbinet_suc] = count_success_non_exact_third_step[idxK, idxT, nn]
                        idx_viterbinet_suc += 1
                if count_success_exact_third_step[idxK, idxT, nn] == 0:
                    if idx_viterbinet_fail < half_n_samples:
                        dataset_observations_fail[idxK, idxT, 0, :, idx_viterbinet_fail] = observations
                        dataset_estimations_fail[idxK, idxT, :, :, idx_viterbinet_fail] = path_trajs2
                        dataset_partial_recovery_fail[idxK, idxT, idx_viterbinet_fail] = count_success_non_exact_third_step[idxK, idxT, nn]
                        idx_viterbinet_fail += 1
                # print('idx_viterbinet_fail', idx_viterbinet_fail)
                if idx_viterbinet_suc >= half_n_samples and idx_viterbinet_fail >= half_n_samples:
                    break
                '''
                # idx_viterbinet_suc        
                ''' 
                in addition to this failure criteria (the gt criteria), add another
                criteria which is: failure iff the true defective set is not in the estimations at all
                '''
                if idx_viterbinet >= n_samples:
                    break
            print('idx_viterbinet_in_estimations', idx_viterbinet_in_estimations)
            print('idx_viterbinet_not_in_estimations', idx_viterbinet_not_in_estimations)
            print('idx_viterbinet', idx_viterbinet)
        elapsed = time.time() - time_start            
        print('It took {:.3f}[min]'.format(elapsed/60))
    # Normalize success and counters
    
    count_success_DD_exact = np.sum(count_success_DD_exact, axis=2) * 100/nmc
    count_success_exact_third_step = np.sum(count_success_exact_third_step, axis=2) * 100/nmc 
    count_success_exact_tot = count_success_DD_exact + count_success_exact_third_step
    count_success_DD_non_exact = np.sum(count_success_DD_non_exact, axis=2) * 100/nmc 
    count_success_non_exact_third_step = np.sum(count_success_non_exact_third_step, axis=2) * 100/nmc 
    count_success_non_exact_tot = count_success_DD_non_exact + count_success_non_exact_third_step

    print('count_success_DD_exact', count_success_DD_exact)
    print('count_success_DD_non_exact', count_success_DD_non_exact)
    
    print('count_success_exact_tot', count_success_exact_tot)
    print('count_success_exact_non_exact_tot', count_success_non_exact_tot)
    count_DND1_avg = np.sum(count_DND1, axis=2) / nmc
    count_PD1_avg = np.sum(count_PD1, axis=2) / nmc
    count_DD2_avg = np.sum(count_DD2, axis=2) / nmc
    count_unknown2_avg = np.sum(count_unknown2, axis=2) / (nmc - count_success_DD_exact*nmc/100) 
    expected_unknown = expected_PD - expected_DD
    count_not_detected = np.matlib.repmat(np.array(vecK), num_of_test_scale,1).T - count_DD2_avg
    if sample_method == 'GE':
        q = ge_model.q
        s = ge_model.s
        pi_B = ge_model.pi_B
        del markov_model
    elif sample_method == 'Markov':
        del ge_model
    if do_third_step and 'viterbi' in third_step_type:
        trans_mat = hmm_model.trans_mat
        init_prob = hmm_model.init_prob
    # Make resutls directory
    results_dir_path = None

    typical_label = '_nottypical'
    if code_type == 'typical_bernoulli':
        typical_label = '_typical'
    
    third_step_label = third_step_type
    if not do_third_step:
        third_step_label = 'None'
    
    permutations_label = ''
    if third_step_type == 'MAP':
        permutations_label = '_perm_factor' + str(permutation_factor) + '_'
    
    viterbi_label = ''
    if third_step_label == 'viterbi+MAP':
        viterbi_label =  '_max_paths_for_lva' + str(max_paths_for_lva) 
        if do_map_if_viterbi_fail:
            viterbi_label = viterbi_label + '_map_if_no_paths'
        else:
            viterbi_label = viterbi_label + '_no_map_if_no_paths'
        viterbi_label = viterbi_label + '_' + str(viterbi_time_steps) + 'steps'

    if save_fig or save_raw:    
        print('Save...')
        time_str = datetime.now().strftime("%d%m%Y_%H%M%S")
        experiment_str = sample_method + '_N' + str(N) + '_K1_' + str(vecK[0]) + '_nmc' + str(nmc) + '_methodDD_' + method_DD + permutations_label + '_thirdStep_' + third_step_label + viterbi_label + '_' + code_type + '_Tbaseline_' +  Tbaseline + '_'
        results_dir_path = os.path.join(save_path, 'countPDandDD_' + experiment_str + time_str)
        os.mkdir(results_dir_path)

# %%

    #%% Visualize
    if is_plot:
        plot_DD_vs_K_and_T(N, vecT, vecK, count_PD1_avg, enlarge_tests_num_by_factors, nmc, count_DD2_avg, sample_method, 
                            method_DD, Tbaseline, code_type, results_dir_path)
        plot_expected_DD(vecK, expected_DD, count_DD2_avg, vecT, enlarge_tests_num_by_factors, results_dir_path)
        plot_expected_PD(vecK, expected_PD, count_PD1_avg, vecT, enlarge_tests_num_by_factors, results_dir_path)
        plot_expected_unknown(vecK, expected_unknown, count_unknown2_avg, vecT, enlarge_tests_num_by_factors, results_dir_path)
        plot_expected_not_detected(vecK, expected_notDetected, count_not_detected, vecT, enlarge_tests_num_by_factors, results_dir_path)
        plot_expected_unknown_avg(vecK, expected_unknown, count_PD1_avg - count_DD2_avg, vecT, 
                                enlarge_tests_num_by_factors, results_dir_path)
        plot_Psuccess_vs_T(vecTs, count_success_DD_exact, count_success_exact_tot, vecK, N, nmc, third_step_label, sample_method, 
                            method_DD, Tbaseline, enlarge_tests_num_by_factors, typical_label, delta_typical_cols,
                            results_dir_path, exact=True)
        plot_Psuccess_vs_T(vecTs, count_success_DD_non_exact, count_success_non_exact_tot, vecK, N, nmc, third_step_label, sample_method, 
                            method_DD, Tbaseline, enlarge_tests_num_by_factors, typical_label,delta_typical_cols,
                            results_dir_path, exact=False)
        
    #%% Save
    if save_raw:
        # fullRawPath = os.path.join(results_dir_path, 'workspace.mat')
        all_variables_names = dir()
        variables_to_save = [var for var in all_variables_names if var not in dont_include_variables]
        save_workspace(results_dir_path, variables_to_save, globals())
        save_code_dir(results_dir_path)

        #%% Save dataset
        # np.savez_compressed(os.path.join(results_dir_path,'dnn_data.npz'), 
        #                     dataset_observations_suc=dataset_observations_suc, 
        #                     dataset_estimations_suc=dataset_estimations_suc,
        #                     dataset_observations_fail=dataset_observations_fail, 
        #                     dataset_estimations_fail=dataset_estimations_fail,
        #                     dataset_partial_recovery_suc=dataset_partial_recovery_suc,
        #                     dataset_partial_recovery_fail=dataset_partial_recovery_fail)
        np.savez_compressed(os.path.join(results_dir_path,'dnn_data.npz'), 
                            dataset_observations=dataset_observations, 
                            dataset_estimations=dataset_estimations,
                            dataset_is_set_in_estimations=dataset_is_set_in_estimations,
                            dataset_is_detected=dataset_is_detected,
                            dataset_true_defective_set=dataset_true_defective_set)
                            # dataset_partial_recovery=dataset_partial_recovery,
                            # dataset_partial_recovery=dataset_partial_recovery)
        
# %%
