import os
import time
from unittest import skip
from tqdm import tqdm
import numpy as np
from datetime import datetime
import itertools
import random
import numpy.matlib
from pyrsistent import v
from sample_population import *
from plotters import *
from calc_bounds_and_num_of_tests import *
import scipy.io

#%% Count #possiblyDefected after CoMa and DD
# 1.
# 1.1. CoMa with T=Tml
# 1.2. count PD1 (should be ~2k)
# 2.
# 2.1. DD 
# 2.2. count PD2
##

#%% Config simulation
# in case we do MAP: if N = 100 => K = 1:7 for enlarge_tests_num_by_factors <= 0.5 (checked)
#                     =if N = 500 => K = ?? (less than 8)
N                   = 500 #100 # TODO:focus on N=500,K=4,5 
vecK                = [10]#[2, 3, 4, 5, 6]
sample_method       = 'GE'  # options: 'ISI', 'onlyPu', 'indicative'
isi_type            = 'asymmetric'
m                   = 1
nmc                 = 50
save_raw            = False
save_fig            = False
save_path           = r'/Users/ayelet/Library/CloudStorage/OneDrive-Technion/Alejandro/count_possibly_defected_results/shelve_raw'
is_plot             = True
do_third_step       = True
is_sort_comb_by_priors = True
add_dd_based_prior  = False
orig_prior_weight   = 0.5
use_typical_codes   = [False] # options: True,False
# ones_zeros_ratio_th = 0.07
delta_typical_cols  = 0.02#0.025 # for N=500,K=10 and T=0.75ML it allows a difference of 2-3 elements in column
delta_typical_rows  = 0.03
enlarge_tests_num_by_factors = [0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.1, 1.2] #[0.25, 0.5, 0.75, 1, 1.25] #[0.5, 0.25, 0.5, 0.75, 1, 1.5]#[0.75, 0.8, 0.9, 1, 1.25, 1.5, 1.75, 2]# [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.3, 1.7, 2] #[0.85, 0.9, 0.95, 1, 1.25, 1.5, 1.75, 2]#
Tbaseline           = 'ML' # options: 'ML', 'lb_no_priors', 'lb_with_priors'
methods_DD          = ['Normal']#{'Normal', 'Sum'} # options: Normal, Iterative, Sum
calc_Pu             = 1
third_step_type     = 'MAP' # options: ['MAP', 'MLE', 'MAP_for_GE']
calc_Pw             = 1
permutation_factor  = 50 # compared [10, 20 , 50, 100] for N=100, K=2,4,6. 50 was the most effective
debug_mode          = False 
check_hamming_dist  = True # count and plot probability of success vs hamming distance in the testing matrix
plot_status_DD      = False
check_var           = True
random.seed(123)
np.random.seed(123)
invalid = -1
all_permutations = []
vecTs = []
for typical_codes in use_typical_codes:
    for method_DD in methods_DD:
        print('===== DD method: {} || typical_codes = {} ====='.format(method_DD, typical_codes))
        ## Initialize counters
        numOfK = len(vecK)
        num_of_test_scale = len(enlarge_tests_num_by_factors)
        count_DND1 = np.zeros((numOfK, num_of_test_scale, nmc))
        count_PD1 = np.zeros((numOfK, num_of_test_scale, nmc))
        count_DD2 = np.zeros((numOfK, num_of_test_scale, nmc))
        count_DND3 = np.zeros((numOfK, num_of_test_scale))
        count_PD3 = np.zeros((numOfK, num_of_test_scale))
        count_unknown2 = np.zeros((numOfK, num_of_test_scale, nmc))
        # count_unknown2_verif = np.zeros((numOfK, num_of_test_scale))
        count_success_DD_exact = np.zeros((numOfK, num_of_test_scale, nmc))
        count_success_DD_non_exact = np.zeros((numOfK, num_of_test_scale, nmc))
        count_add_success_third_step = np.zeros((numOfK, num_of_test_scale, nmc))
        count_success_non_exact_third_step = np.zeros((numOfK, num_of_test_scale, nmc))
        count_not_detected = np.zeros((numOfK, num_of_test_scale))
        expected_notDetected = np.zeros((numOfK, num_of_test_scale))
        expected_DD = np.zeros((numOfK, num_of_test_scale))
        expected_PD = np.zeros((numOfK, num_of_test_scale))
        expected_unknown = np.zeros((numOfK, num_of_test_scale))
        iter_until_detection_tot = np.zeros((numOfK, num_of_test_scale))
        iter_until_detection_CoMa_and_DD = np.zeros((numOfK, num_of_test_scale))
        iter_until_detection_third_step_eff = np.zeros((numOfK, num_of_test_scale)) # effective 
        iter_until_detection_third_step_full = np.zeros((numOfK, num_of_test_scale)) # if we don't stop map or ml after the first match Y=Yw
        queries_per_third_step_iter = 0
        Pw_of_true_out_of_max_Pw = np.zeros((numOfK, num_of_test_scale, nmc))
        correct_Pw = np.zeros((numOfK, num_of_test_scale, nmc))
        correctPw_outof_maxPw = np.zeros((numOfK, num_of_test_scale, nmc))
        if check_hamming_dist:
            hamming_dist_avg_vec = np.zeros((numOfK, num_of_test_scale, nmc))
            hamming_dist_min_vec = np.zeros((numOfK, num_of_test_scale, nmc))
        # check_var_DD = np.zeros((numOfK, num_of_test_scale, nmc))
        # check_var_unknowns = np.zeros((numOfK, num_of_test_scale, nmc))
        
        #%% Start simulation
        for idxK in range(numOfK):
            K = vecK[idxK]
            print('K = ' + str(K))
            # For each K calculate number of test according the Tml and scale factor
            ge_model = None
            if sample_method == 'GE': # create the ge model 
                _, ge_model = sample_population_gilbert_elliot_channel(N, K, ge_model)
            vecT = calculate_vecT_for_K(K, N, enlarge_tests_num_by_factors, Tbaseline=Tbaseline, Pe=Pe, 
                        sample_method=sample_method, ge_model=ge_model, Pu=None, coeff_mat=None)
            vecTs.append(vecT)
            time_start = time.time()
            for idxT in range(num_of_test_scale):
                T = np.int16(vecT[idxT])
                print('T', T)
                overflow_const = 1 #10^T

                p = 1-2**(-1/K) # options: 1/K, log(2)/K, 1-2**(-1/K)
                expected_PD[idxK, idxT] = K + (N-K) * (1-p*(1-p)**K)**T 
                nPD = expected_PD[idxK, idxT]
                expected_DD[idxK, idxT]  = K*(1-(1-p*(1-p)**(nPD-1))**T)#nPD*p_defective*(1-(1-p*(1-p)**(nPD-1))**T) # version1 - p_defective appears once
                expected_notDetected[idxK, idxT] = K - expected_DD[idxK, idxT]
                count_success_DD_exact_vec_nmc = np.zeros((nmc,))
                count_success_DD_non_exact_vec_nmc = np.zeros((nmc,))
                if check_hamming_dist:
                    # hamming_dist_avg_vec = np.zeros((nmc,))
                    # hamming_dist_min_vec = np.zeros((nmc,))
                    count_PD_nn = np.zeros((nmc,))
                    count_DD_nn = np.zeros((nmc,))
                    count_unknowns_nn = np.zeros((nmc,))
                    min_ones_ratio_in_X = np.zeros((nmc,))
                    max_ones_ratio_in_X = np.zeros((nmc,))
                    sum_X_nn = np.zeros((nmc,))
                    sum_col_in_X_max_nn = np.zeros((nmc,))
                    list_good_X_occluded = []
                    list_bad_X_occluded = []
                for nn in tqdm(range(nmc), desc='T ' + str(T)):
                # for nn in range(nmc):
                    # print('nn', nn)
                    ## Sample
                    if sample_method == 'ISI':
                        if m == 1:
                            U, Pu, coeff_mat = sample_population_ISI_m1(N, K)
                        else:
                            U, W, Pu, coeff_mat, Pw = sample_population_ISI(N, K, m, all_permutations, isi_type, calc_Pw, calc_Pu)
                    elif sample_method == 'onlyPu':
                        U, Pu, Pw = sample_population_no_corr(N, K)
                    elif sample_method == 'indicative':
                        U, Pu, Pw, num_of_distractions = sample_population_indicative(N, K)
                    elif sample_method == 'GE':
                        U, ge_model = sample_population_gilbert_elliot_channel(N, K, ge_model)
                    true_defective_set = np.where(U == 1)[1].tolist()
                    ## 1. Definitely Not Defective
                    # Encoder - bernoulli 
                    X =  np.multiply(np.random.uniform(0,1,(T, N)) < p,1) # iid testing matrix
                    
                    '''typicality in cols
                    if typical_codes:
                        non_typical_cols = np.arange(N)
                        while len(non_typical_cols)>0:
                            # randomize new rows and insert in X
                            num_of_non_typical_cols = len(non_typical_cols)
                            newCols = np.random.uniform(0,1,(T, num_of_non_typical_cols)) < p
                            X[:, non_typical_cols] = newCols
                            # find nontypical rows
                            number_of_ones_in_X = np.sum(X[:,non_typical_cols],0)
                            number_of_zeros_in_X = N-number_of_ones_in_X
                            non_typical_cols = np.where(abs(number_of_ones_in_X / N - p) > ones_zeros_ratio_th)[0]
                            if check_hamming_dist:
                                min_ones_ratio_in_X[nn] = np.min(abs(number_of_ones_in_X / N - p))
                                max_ones_ratio_in_X[nn] = np.max(abs(number_of_ones_in_X / N - p))
                    
                    ## typicality in all the matrix, not very tight threshold
                    if typical_codes:
                        # non_typical_cols = np.arange(N)
                        # while len(non_typical_cols)>0:
                            # randomize new rows and insert in X
                        # num_of_non_typical_cols = len(non_typical_cols)
                        typical = False
                        while not typical:
                            X = np.multiply(np.random.uniform(0,1,(T, N)) < p,1)
                            sum_rows = np.sum(X, axis=1)
                            sum_cols = np.sum(X, axis=0)
                            dist_rows = np.abs(sum_rows / N - p)
                            dist_cols = np.abs(sum_cols / T - p)
                            non_typical_row = [r for r in dist_rows if r > ones_zeros_ratio_th]
                            non_typical_col = [c for c in dist_cols if c > ones_zeros_ratio_th]
                            if non_typical_row or non_typical_col:
                                typical = False
                            else:
                                typical = True
                    '''
                    if typical_codes:
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

                    tested_mat = X*U
                    Y = np.sum(tested_mat, 1) > 0 

                    if check_hamming_dist:
                        hamming_dist_mat = compute_HammingDistance(X.T)
                        lower_triangle_hamming = np.tril(hamming_dist_mat)
                        hamming_dist_avg_vec[idxK, idxT, nn] = np.sum(lower_triangle_hamming)/(N*(N-1))
                        hamming_dist_min_vec[idxK, idxT, nn] = np.min(lower_triangle_hamming[np.nonzero(lower_triangle_hamming)])
                        man_min_hamming_dist = N
                    
                    X_mark_occlusion = np.zeros((X.shape[0]+1, X.shape[1]))
                    X_mark_occlusion[1:, :] = X
                    X_mark_occlusion[0,true_defective_set] = 5 
                    if check_hamming_dist:
                        sum_X_nn[nn] = np.sum(X)
                        sum_col_in_X_max_nn[nn] = np.mean(np.sum(X,axis=0))
                    
                    # Decoder - CoMa
                    PD1 = np.arange(N)
                    DND1 = []
                    for ii in range(T):
                        if len(PD1) <= K:
                            break 
                        if Y[ii] == 0:
                            for jj in PD1:
                                iter_until_detection_CoMa_and_DD[idxK, idxT] += 1
                                if X[ii,jj] == 1: # definitely not defected
                                    PD1 = PD1[PD1 != jj]
                                    DND1 += [jj]
                    count_DND1[idxK, idxT, nn] = len(DND1)
                    count_PD1[idxK, idxT, nn] = len(PD1)
                    if check_hamming_dist:
                            count_PD_nn[nn] = len(PD1)
                    if len(PD1) <= K: # all the PD are DD - all defective found
                        count_DD2[idxK, idxT, nn] = len(PD1)
                        # if check_var:
                            # check_var_DD[idxK, idxT, nn] = len(DD2)
                        count_success_DD_exact[idxK, idxT, nn] += 1
                        if check_hamming_dist:
                            count_success_DD_exact_vec_nmc[nn] = 1
                            count_success_DD_non_exact_vec_nmc[nn] = 1
                            count_DD_nn[nn] = K
                            count_unknowns_nn[nn] = 0
                        count_success_DD_non_exact[idxK, idxT, nn] += 1
                        continue
                    ## 2. Definite Defective
                    # steps 1&2
                    if method_DD == 'Normal':
                        DD2 = []
                        for ii in range(T):
                            if Y[ii] == 1 and np.sum(X[ii,PD1]) == 1: # only 1 item among the PD equals 1 and the rest equal 0
                                jj = np.where(X[ii,PD1] == 1)[0][0] # find the definite defective item index in PD1 array
                                iter_until_detection_CoMa_and_DD[idxK, idxT] += len(PD1)

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
                        # if check_var:
                            # check_var_DD[idxK, idxT, nn] = len(DD2)
                        
                        if len(DD2) >= K: # all defective found
                            count_success_DD_exact[idxK, idxT, nn] += 1
                            if check_hamming_dist:
                                count_success_DD_exact_vec_nmc[nn] = 1
                                count_success_DD_non_exact_vec_nmc[nn] = 1
                                count_DD_nn[nn] = len(DD2)
                                count_unknowns_nn[nn] = 0
                            count_success_DD_non_exact[idxK, idxT, nn] +=1
                            continue
                        num_of_false_positive_in_DD2 = [e for e in DD2 if e not in true_defective_set]
                        if num_of_false_positive_in_DD2:
                            print('Something wrong with the DD - not defective detected')
                            pass
                    ## 2. TRY iterative Definite Defective - currently there is a bug:

                    # I need to see in my notes what we said about participating
                    # item and sum=1
                    # steps 1&2
                    elif method_DD == 'Iterative':
                        DD2 = []
                        try_again_DD = True
                        while try_again_DD:
                            try_again_DD = False
                            for tt in range(T):
                                if Y[tt] == 0:
                                    continue
                                # calculate sum over not DD
                                count_participant_who_are_not_DD = 0
                                participants = np.where(X[tt,PD1] == 1)[0] # the indices of the PD that participate int the ii test
                                                                    # the corresponding index in X = PD1(participants)
                                participants_who_are_not_DD = [e for e in PD1[participants] if e not in DD2]
                                count_participant_who_are_not_DD = len(participants_who_are_not_DD)
                                if count_participant_who_are_not_DD == 1: #&& participants_who_are_not_DD # only 1 item among the PD (who have not already been detected as DD) equals 1 and the rest equal 0
                                    defective = participants(participants_who_are_not_DD) # index of the defective in X (index between 1 to N)
                                    if defective not in DD2: # add jj only if jj is not already detected as DD
                                        DD2 += [defective]
                                        try_again_DD = True
                                    
                        count_DD2[idxK, idxT, nn] = len(DD2)
                        # if check_var:
                        #     check_var_DD[idxK, idxT, nn] = len(DD2)
                        if len(DD2) >= K: # all defective found
        #                     fprintf('All defective found\n')
                            count_success_DD_exact[idxK, idxT, nn] += 1
                            if check_hamming_dist:
                                count_success_DD_exact_vec_nmc[nn] = 1
                                count_success_DD_non_exact_vec_nmc[nn] = 1
                                count_DD_nn[nn] = len(DD2)
                                count_unknowns_nn[nn] = 0
                            count_success_DD_non_exact[idxK, idxT, nn] += 1
                            continue 
                    ##
                    elif method_DD == 'Sum':
                        Y_sum = np.sum(tested_mat, 1) # keep the levels
                        iter = 0
                        helpful_iter = []
                        DD2 = []
                        DD_rows = []
                        try_again_DD = True
                        while try_again_DD:
                            try_again_DD = 0
                            iter = iter + 1
                            for tt in range(T):
                                if Y[tt] == 0 or tt in DD_rows:# skip rows that based on them we detected DD
                                    continue
                                count_participant_who_are_not_DD = 0
                                participants = np.where(X[tt,PD1] == 1)[0] # the indices of the PD that participate int the ii test
                                                                    # the corresponding index in X = PD1(participants)
                                participants_who_are_not_DD = [e for e in PD1[participants] if e not in DD2]
                                count_participant_who_are_not_DD = len(participants_who_are_not_DD)
                                participants_who_are_DD = [e for e in PD1[participants] if e in DD2]
                                count_participant_who_are_DD = len(participants_who_are_DD)
                                if count_participant_who_are_not_DD == 1 and Y_sum[tt]-count_participant_who_are_DD == 1: #&& participants_who_are_not_DD # only 1 item among the PD (who have not already been detected as DD) equals 1 and the rest equal 0
                                    defective = participants_who_are_not_DD[0]#index of the defective in X (index between 1 to N)
                                    if defective not in DD2: # add jj only if jj is not already detected as DD
                                        DD2 += [defective]
                                        DD_rows += [tt]
                                        try_again_DD = True
                                        helpful_iter += [iter]
                                    
                        count_DD2[idxK, idxT, nn] = len(DD2)
                        # if check_var:
                        #     check_var_DD[idxK, idxT, nn] = len(DD2)
                        
                        if len(DD2) >= K: # all defective found
                            count_success_DD_exact[idxK, idxT, nn] += 1
                            if check_hamming_dist: 
                                count_success_DD_exact_vec_nmc[nn] = 1
                                count_success_DD_non_exact_vec_nmc[nn] = 1
                                count_DD_nn[nn] = len(DD2)
                                count_unknowns_nn[nn] = 0
                            count_success_DD_non_exact[idxK, idxT, nn] += 1
                            continue 
        
                    # find all unknown
                    unknown2 = [e for e in PD1 if e not in DD2]#PD1[PD1 not in DD2][0]
                    
                    count_unknown2[idxK, idxT, nn] = len(unknown2)

                    if check_hamming_dist:
                        print('#unknown = {}\t E[unknowns] = {}\t min_hamming_dist = {}'.format(len(unknown2), expected_PD[idxK, idxT]-expected_DD[idxK, idxT], hamming_dist_min_vec[idxK, idxT, nn]))
                        print('#DD = {}\t \t E[DD] = {}'.format(len(DD2), expected_DD[idxK, idxT]))
                        count_DD_nn[nn] = len(DD2)
                        count_unknowns_nn[nn] = len(unknown2)
                        num_of_permutations_binomial = math.comb(len(unknown2), K-len(DD2))
                        print('num_of_permutations_binomial', num_of_permutations_binomial)
                        # if len(unknown2) > 30:
                        # plt.imshow(X)
                        # plt.title('bad' if len(unknown2) - len(unknown2) > 32 else 'good')
                        # plt.show()
                        if True:
                            if num_of_permutations_binomial > 5e6:# num_of_permutations_binomial > 5e6
                                if len(list_bad_X_occluded) < 3:
                                    list_bad_X_occluded.append(X_mark_occlusion)
                            elif len(list_good_X_occluded) < 3:
                                list_good_X_occluded.append(X_mark_occlusion)
                            if len(list_good_X_occluded) >= 3 and len(list_bad_X_occluded) >= 3:
                                dic = {'good1': list_good_X_occluded[0], 
                                        'good2':list_good_X_occluded[1], 
                                        'good3': list_good_X_occluded[2],
                                        'bad1': list_bad_X_occluded[0], 
                                        'bad2':list_bad_X_occluded[1], 
                                        'bad3': list_bad_X_occluded[2]}
                                scipy.io.savemat(r'/Users/ayelet/Library/CloudStorage/OneDrive-Technion/Alejandro/temp_res/occlusions.mat', dic)       
                                pass
                    # print('#PD = {} || #DD = {} || #unknown = {}'.format(len(PD1), len(DD2), len(unknown2)))
                    # print('hamming_dist_min', hamming_dist_min)
                    # if check_var:
                    #     check_var_unknowns[idxK, idxT, nn] = len(unknown2)
                    # count_unknown2_verif[idxK, idxT] += len(PD1) - len(DD2)
                    count_not_detected_defectives = K-len(DD2)
                    count_success_DD_non_exact[idxK, idxT, nn] += (len(DD2) / K)
                    if check_hamming_dist:
                        count_success_DD_non_exact_vec_nmc[nn] = (len(DD2) / K)
                        # X_mark = mark_hidden_in_DD(X, save_path=r'/Users/ayelet/Library/CloudStorage/OneDrive-Technion/Alejandro/temp_res/')
                    ## 3.
                    if is_plot and plot_status_DD:
                        plot_status_before_third_step(N, K, T, enlarge_tests_num_by_factors[idxT], PD1, DD2, true_defective_set) 
                    if not do_third_step:
                        continue

                    if third_step_type == 'MAP':
                        if add_dd_based_prior: 
                            # 3.1 calculate new priors based on X(Y==1, PD1) (after DD)
                            Pu_DD2 = np.zeros((1, len(PD1)))
                            for tt in range(T):
                                if Y[tt] == 0:
                                    continue
                                PD_participants = np.where(X[tt,PD1] == 1)[0] # participants in 1,...,#PD indices
                                participants_who_are_DD = [e for e in PD1[PD_participants] if e in DD2]

                                if len(participants_who_are_DD) > 0:
                                    # is there something I can deduce if there is a DD in the line?
                                    continue
                                nParticipants = len(PD_participants)
                                Pu_DD2[0, PD_participants] = Pu_DD2[0, PD_participants] + 1/nParticipants
                                ''' 
                                This new prior Pu_DD2 may suggest that a really defective item has prior Pu=0
                                This may happen if there are 2 defective (or more) in one row
                                That's why when we want to merge the priors we don't put too much weight on Pu_DD2=0.
                                However, if the original given prior says Pu = 0, then we keep that 0
                                '''
                        
                        if sample_method == 'GE':

                            if add_dd_based_prior: 
                                pass

                            if is_sort_comb_by_priors:
                                # print('sort permutations')
                                # all_permutations3, Pw_sorted, Pu3 = ge_model.sort_comb_by_priors_GE(N, all_permutations3, DD2, DND1)
                                all_permutations3, Pw_sorted, num_of_iterations_in_sort = ge_model.sort_comb_by_priors_GE_cut_by_entropy(N, K, T, nPD, DD2, DND1, unknown2, permutation_factor=permutation_factor)
                                iter_until_detection_third_step_full[idxK, idxT] += num_of_iterations_in_sort
                                iter_until_detection_third_step_eff[idxK, idxT] += num_of_iterations_in_sort
                            pass
                        else:
                            if add_dd_based_prior: 
                                # use weighted average to merge the 2 probabilities
                                # where one of the 2 prob is 0, keep the merged probability as zero
                                zero_probability_idx_Pu = np.where(Pu[:,PD1] == 0)[1]
                                zero_probability_idx_Pu_DD2 = np.where(Pu_DD2 == 0)[1]

                                Pu3 = orig_prior_weight * Pu[:,PD1] + (1-orig_prior_weight) * Pu_DD2
                                Pu3[0, zero_probability_idx_Pu.tolist()] = 0

                            else:
                                Pu3 = Pu[:,PD1].copy()
                            # item detected as defective gets prob 1 to be defective in this step:
                            idx_of_DD_in_PD_coor = [list(PD1).index(item) for item in DD2]
                            Pu3[0,idx_of_DD_in_PD_coor] = 1                        
                        
                            if is_sort_comb_by_priors:
                                Pu3_all_population = np.zeros((1, N))
                                Pu3_all_population[0,PD1] = Pu3
                                Pu3 = Pu3_all_population
                                Pu3 = np.matmul(Pu3,coeff_mat) # not only initial prbabilitis but 
                                                    # also taking in account the correlation
                                if sample_method == 'ISI':
                                    if m==1:
                                        debug_correct_permute = np.where(U==1)[1]
                                        # TODO: calculate Pu3 again. if item in DD2 so Pu3=1, and 
                                        # 1. switch Pu = 1 for each item in DD2 
                                        # 2. Pu3(Uj) = Pu0 @ coeffmat
                                        all_permutations3, Pw_sorted = sort_comb_by_priors_ISI_m1(all_permutations3, Pu3, coeff_mat, DD2, debug_correct_permute)
                                    else:
                                        pass
                                else:
                                    all_permutations3, Pw_sorted = sort_comb_by_priors(all_permutations3, Pu3)
                                
                        num_of_permutations3 = all_permutations3.shape[0]
                        ''' 
                        check what is Pw of the true permute - what is the rate of this Pw  out of the max Pw.. 
                        
                        for ii, permute in enumerate(all_permutations3):
                            if sorted(list(permute)+DD2) == sorted(list(np.where(U==1)[1])):
                                correct_permutation = ii
                                correct_Pw[idxK, idxT, nn] = Pw_sorted[ii]
                                correctPw_outof_maxPw[idxK, idxT, nn] = Pw_sorted[ii]/Pw_sorted[0]
                                pass
                        '''
                        apriori = invalid*np.ones((Pw_sorted.shape[0],1))
                        item_permute = 0
                        valid_options = 0
                        # print('num_of_permutations3', num_of_permutations3)
                        for comb in range(Pw_sorted.shape[0]):                            
                            if not debug_mode:
                                if Pw_sorted[comb] / Pw_sorted[0] < 0.1 and valid_options >=1: # the rate was bigger than 0.6 in sim for the true defective set
                                    break
                            item_permute += 1
                            # calculate Y for the w-th permutation 
                            # print('all_permutations3.shape', all_permutations3.shape)
                            # print('all_permutations3', all_permutations3)
                            permute = all_permutations3[comb,:].tolist()

                            U_forW = np.zeros((1,N))
                            U_forW[0,permute + DD2] = 1
                            
                            X_forW = X*U_forW
                            Y_forW = np.sum(X_forW, 1) > 0

                            # possible case: Yw = Y
                            # the case: Yw = 0 and Y = 1 is possible when T is too small
                            # for example: U = [1 0 0] T=1 X = [0 0 0]

                            ## skip the probailities calculation and comparison
                            #  just take the first premute w that satisfy Y==Yw
                            # [the probailities are already sorted from high to low]
                            
                            # skip the case: Yw = 1 and Y = 0:
                            if (Y_forW != Y).any():
                                if debug_mode and set(permute+DD2) == set(true_defective_set):
                                    print('Yw!=Y')
                                continue
                            valid_options += 1
                            apriori[comb] = 1e16#1e32 #overflow_const
                            if sample_method == 'ISI':
                                apriori[comb] *= Pw_sorted[comb] * p ** np.sum(X_forW == 1)
                            elif sample_method == 'GE':
                                # calc P(w|Xsw) 
                                # TODO: check why not using Pw_sorted?
                                Pw_by_Xsw = 1e16
                                for tt in range(T):
                                    participating_items = np.where(X_forW[tt,:] == 1)[0]
                                    if participating_items.shape[0] == 0:
                                        continue

                                    probability_per_item = [ge_model.get_conditional_probability_GE(item, DD2, DND1) for item in participating_items]
                                    Pw_by_Xsw *= np.prod(probability_per_item)
                                    # Pw_by_Xsw *= np.sum(probability_per_item)
                                
                                apriori[comb] = Pw_by_Xsw * p ** np.sum(X_forW == 1)
                                
                            else:
                                for tt in range(T):
                                    num_of_ones_in_Xwt = np.sum(X_forW[tt,:] == 1)
                                    P_q = overflow_const * p**num_of_ones_in_Xwt
                                    P_Xsw_t = P_q
                                    for ii in permute: 
                                        P_Xsw_t *= Pu3[0,ii]
                                    apriori[comb] *= P_Xsw_t # get zeros
                            
                            if set(permute+DD2) == set(true_defective_set) and debug_mode:
                                print('true defective set prior: prior(W*) = ' + str(apriori[comb,0]))
                        iter_until_detection_third_step_full[idxK, idxT] += Pw_sorted.shape[0]
                        max_likelihood_W = np.argmax(apriori)
                        if debug_mode:
                            print('chosen defective set prior: prior(estW) = ' + str(apriori[max_likelihood_W,0]))
                        estU = np.zeros(U.shape)
                        estU[0, all_permutations3[max_likelihood_W,:]] = 1  
                        estU[0, DD2] = 1
                        iter_until_detection_third_step_eff[idxK, idxT] += max_likelihood_W
                        Pw_of_true_out_of_max_Pw[idxK, idxT, nn] = Pw_sorted[max_likelihood_W] / Pw_sorted[0]
                        
                    elif third_step_type == 'MAP_for_GE':
                        all_permutations3, Pw_sorted, num_of_iterations_in_sort = ge_model.sort_comb_by_priors_GE_cut_by_entropy(K, T, nPD, DD2, DND1, unknown2, permutation_factor=permutation_factor)
                        iter_until_detection_third_step_full[idxK, idxT] += num_of_iterations_in_sort
                        iter_until_detection_third_step_eff[idxK, idxT] += num_of_iterations_in_sort
                        num_of_permutations3 = all_permutations3.shape[0]
                        apriori = invalid*np.ones((Pw_sorted.shape[0],1))
                        # print('num_of_permutations3', num_of_permutations3)
                        for comb in range(Pw_sorted.shape[0]):                            
                            permute = all_permutations3[comb,:].tolist()
                            U_forW = np.zeros((1,N))
                            U_forW[0,permute + DD2] = 1
                            X_forW = X*U_forW
                            Y_forW = np.sum(X_forW, 1) > 0

                            # possible case: Yw = Y
                            # the case: Yw = 0 and Y = 1 is possible when T is too small
                            # for example: U = [1 0 0] T=1 X = [0 0 0]

                            ## skip the probailities calculation and comparison
                            #  just take the first premute w that satisfy Y==Yw
                            # [the probailities are already sorted from high to low]
                            
                            # skip the case: Yw = 1 and Y = 0:
                            if (Y_forW != Y).any():
                                if debug_mode and set(permute+DD2) == set(true_defective_set):
                                    print('Yw!=Y')
                                continue
                            
                            # apriori[comb] = 1e16#1e32 #prevent overflow_const
                            Pw_by_Xsw = 1e16
                            for tt in range(T):
                                participating_items = np.where(X_forW[tt,:] == 1)[0]
                                if participating_items.shape[0] == 0:
                                    continue

                                probability_per_item = [ge_model.get_conditional_probability_GE(item, DD2, DND1) for item in participating_items]
                                Pw_by_Xsw *= np.prod(probability_per_item)
                            apriori[comb] = Pw_by_Xsw * p ** np.sum(X_forW == 1)
                            # print('p ** np.sum(X_forW == 1)', p ** np.sum(X_forW == 1))
                            if set(permute+DD2) == set(true_defective_set) and debug_mode:
                                print('true defective set prior: prior(W*) = ' + str(apriori[comb,0]))
                        iter_until_detection_third_step_full[idxK, idxT] += Pw_sorted.shape[0]
                        max_likelihood_W = np.argmax(apriori)
                        if debug_mode:
                            print('chosen defective set prior: prior(estW) = ' + str(apriori[max_likelihood_W,0]))
                        estU = np.zeros(U.shape)
                        estU[0, all_permutations3[max_likelihood_W,:]] = 1  
                        estU[0, DD2] = 1
                        iter_until_detection_third_step_eff[idxK, idxT] += max_likelihood_W
                        Pw_of_true_out_of_max_Pw[idxK, idxT, nn] = Pw_sorted[max_likelihood_W] / Pw_sorted[0]

                    elif third_step_type == 'MLE':
                        try:
                            all_permutations3_no_prior = np.array(list(itertools.combinations(unknown2, count_not_detected_defectives)))
                            num_of_all_permutations3_no_prior = all_permutations3_no_prior.shape[0]
                        except:
                            print('could not find permutations')
                            continue
                        # we want to find estW: estW = argmax{P(Y|W)}
                        # mle_error_counter = np.zeros((1,num_of_all_permutations3_no_prior))
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
                        # if max_likelihood_W is None:
                        #     max_likelihood_W = np.argmin(mle_error_counter)
                        iter_until_detection_third_step_eff[idxK, idxT] =+ w
                        iter_until_detection_third_step_full[idxK, idxT] =+ all_permutations3_no_prior.shape[0]
                        estU = np.zeros(U.shape)
                        estU[0, all_permutations3_no_prior[max_likelihood_W,:]] = 1  
                        estU[0, DD2] = 1
                        
                    if np.sum(U != estU) == 0:
                        count_add_success_third_step[idxK, idxT, nn] += 1
                        count_success_non_exact_third_step[idxK, idxT, nn] += (K-len(DD2))/K
                    else:
                        # count only the items detected in the 3rd step 
                        detected_defectives = np.where(estU==1)[1] # may be errornous detection
                        not_detected = set(true_defective_set)-set(detected_defectives)
                        num_of_correct_detection = K-len(not_detected)
                        count_success_non_exact_third_step[idxK, idxT, nn] += (num_of_correct_detection-len(DD2))/K 
                if False: #check_hamming_dist:
                    debug_info_df = pd.DataFrame({'PD': count_PD_nn, 
                                                'DD': count_DD_nn, 
                                                'unknowns': count_unknowns_nn, 
                                                'hamming_dist_min':hamming_dist_min_vec,
                                                'hamming_dist_avg':hamming_dist_avg_vec,
                                                'min_ones_ratio_in_X': min_ones_ratio_in_X,
                                                'max_ones_ratio_in_X': max_ones_ratio_in_X,
                                                'sum_X_nn': sum_X_nn,
                                                'sum_col_in_X_max_nn': sum_col_in_X_max_nn})
                    
                    fig = px.scatter(debug_info_df, x="hamming_dist_min", y="unknowns")
                    fig.show()
                    fig = px.scatter(debug_info_df, x="hamming_dist_min", y="DD")
                    fig.show()
                    fig = px.scatter(debug_info_df, x="max_ones_ratio_in_X", y="unknowns")
                    fig.show()
                    fig = px.scatter(debug_info_df, x="max_ones_ratio_in_X", y="DD")
                    fig.show()
                    fig = px.scatter(debug_info_df, x="sum_X_nn", y="unknowns")
                    fig.show()
                    fig = px.scatter(debug_info_df, x="sum_X_nn", y="DD")
                    fig.show()
                    fig = px.scatter(debug_info_df, x="sum_col_in_X_max_nn", y="DD")
                    fig.show()
                    fig = px.scatter(debug_info_df, x="sum_col_in_X_max_nn", y="unknowns")
                    fig.show()
                if False: #is_plot and check_hamming_dist:
                    # plot_DD_exact_Ps_vs_min_and_avg_hamming_dist(N, K, T, enlarge_tests_num_by_factors[idxT], count_success_DD_exact_vec_nmc, hamming_dist_avg_vec, hamming_dist_min_vec)
                    plot_DD_non_exact_Ps_vs_min_and_avg_hamming_dist(N, K, T, enlarge_tests_num_by_factors[idxT], count_success_DD_non_exact_vec_nmc, hamming_dist_avg_vec, hamming_dist_min_vec)

            elapsed = time.time() - time_start            
            print('It took {:.3f}[min]'.format(elapsed/60))
        # Normalize success and counters
        
        count_success_DD_exact = np.sum(count_success_DD_exact, axis=2) * 100/nmc
        count_add_success_third_step = np.sum(count_add_success_third_step, axis=2) * 100/nmc 
        count_success_exact_tot = count_success_DD_exact + count_add_success_third_step
        count_success_DD_non_exact = np.sum(count_success_DD_non_exact, axis=2) * 100/nmc 
        count_success_non_exact_third_step = np.sum(count_success_non_exact_third_step, axis=2) * 100/nmc 
        count_success_non_exact_tot = count_success_DD_non_exact + count_success_non_exact_third_step

        iter_until_detection_CoMa_and_DD /= nmc
        iter_until_detection_third_step_eff /= nmc
        iter_until_detection_third_step_full /= nmc
        iter_until_detection_tot = iter_until_detection_CoMa_and_DD + iter_until_detection_third_step_full
        print('count_success_exact_tot', count_success_exact_tot)
        print('count_success_exact_non_exact_tot', count_success_non_exact_tot)
        print('iter_until_detection_tot', iter_until_detection_tot)
        count_DND1_avg = np.sum(count_DND1, axis=2) / nmc
        count_PD1_avg = np.sum(count_PD1, axis=2) / nmc
        count_DD2_avg = np.sum(count_DD2, axis=2) / nmc
        count_unknown2_avg = np.sum(count_unknown2, axis=2) / (nmc - count_success_DD_exact*nmc/100) 
        # count_unknown2_verif = count_unknown2_verif / (nmc - count_success_DD_exact) 
        expected_unknown = expected_PD - expected_DD
        count_not_detected = np.matlib.repmat(np.array(vecK), num_of_test_scale,1).T - count_DD2_avg
        
        # Make resutls directory
        results_dir_path = None

        typical_label = '_nottypical'
        if typical_codes:
            typical_label = '_typical'
        
        third_step_label = third_step_type
        if not do_third_step:
            third_step_label = 'None'
        
        permutations_label = ''
        if third_step_type == 'MAP':
            permutations_label = '_perm_factor' + str(permutation_factor) + '_'

        if save_fig or save_raw:    
            time_str = datetime.now().strftime("%d%m%Y_%H%M%S")
            experiment_str = 'N' + str(N) + '_nmc' + str(nmc) + '_methodDD_' + method_DD + permutations_label + '_thirdStep_' + third_step_label + typical_label + '_Tbaseline_' +  Tbaseline + '_'
            results_dir_path = os.path.join(save_path, 'countPDandDD_' + experiment_str + time_str)
            os.mkdir(results_dir_path)
        
        #%% Visualize
        if is_plot:
            plot_DD_vs_K_and_T(N, vecT, vecK, count_PD1_avg, enlarge_tests_num_by_factors, nmc, count_DD2_avg, sample_method, 
                                method_DD, Tbaseline, typical_codes, results_dir_path)
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
            fullRawPath = os.path.join(results_dir_path, 'workspace.mat')
            all_variables_names = dir()
            # remove packages(numpy), functions(calculatePu), set type('not_detected'),... 
            dont_include_variables = ['np', 'scipy', 'scipy.io', 'numpy', 'pd', 'matplotlib', \
                                    'time', 'tqdm', 'math', 'itertools', 'random', 'go', 'px', \
                                    'datetime', 'os', 'plt', 'shelve', 'reverse', \
                                    'plot_DD_vs_K_and_T', 'plot_expected_DD', 'plot_expected_PD', 'plot_expected_unknown', \
                                    'plot_expected_not_detected', 'plot_expected_unknown_avg', 'plot_Psuccess_vs_T', 'plot_and_save', \
                                    'save_workspace', 'load_workspace', 'rand_array_fixed_sum', 'split_list_into_2_sequence', \
                                    'sample_population_no_corr', 'sample_population_ISI', 'sample_population_ISI+m1', \
                                    'spread_infection_using_corr_mat', \
                                    'calculatePu', 'calculatePw', 'test_sample_population_no_corr', 'test_sample_population_ISI', \
                                    'test_sample_population_ISI_m1', 'sample_population_indicative', 'sort_comb_by_priors', \
                                    'sort_comb_by_priors_ISI_m1' , \
                                    'calculate_lower_bound_ISI_m1', 'calc_entropy_y_given_x_binary_RV', 'calc_entropy_binary_RV', \
                                    'test_calculate_lower_bound_ISI_m1', \
                                    'not_detected', 'ge_model', 'perm', 'combinations', 'permutations'\
                                    'num_of_false_positive_in_DD2', 'enlarge_tests_num_by_factors', 'count_add_success_third_step', 'count_not_detected_defectives', \
                                    '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__','__spec__', 'fig']
            # dont_include_variables.append(dir(plotters), dir(sample_population), dir(calc_bounds_and_num_of_tests))
            variables_to_save = [var for var in all_variables_names if var not in dont_include_variables]
            save_workspace(fullRawPath, variables_to_save, globals())

            

# %%
