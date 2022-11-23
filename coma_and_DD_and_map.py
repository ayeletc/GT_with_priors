import os
import numpy as np
from datetime import datetime
import itertools
import random
import numpy.matlib
from pyrsistent import v
from sample_population import *
from plotters import *
from calc_bounds_and_num_of_tests import *

#%% Count #possiblyDefected after CoMa and DD
# 1.
# 1.1. CoMa with T=Tml
# 1.2. count PD1 (should be ~2k)
# 2.
# 2.1. DD 
# 2.2. count PD2
##
    
#%% Config simulation
# in case we do MAP: if N = 100 => K = 1:7 for enlarge_tests_num_by_factors ≤ 0.5 (checked)
#                     =if N = 500 => K = ?? (less than 8)
N                   = 100
vecK                = [2, 5, 10, 20, 30]#[10, 20, 30, 40]#[10, 50, 100, 150, 200, 250, 300, 350]#[5,10,20,30,40,50] #np.arange(2,50,5)#np.arange(2,20,2)#1:5:30#1:20:150 #10#round(beta * N ^ alpha)
sample_method       = 'GE'  # options: 'ISI', 'onlyPu', 'indicative'
isi_type            = 'asymmetric'
m                   = 1
nmc                 = 200
save_raw            = True
save_fig            = True
save_path           = r'/Users/ayelet/Library/CloudStorage/OneDrive-Technion/Alejandro/count_possibly_defected_results/shelve_raw'
is_plot             = True
do_third_step       = True
is_sort_comb_by_priors = True
add_dd_based_prior  = False
orig_prior_weight   = 0.5
use_typical_codes   = [True] # options: True,False
ones_zeros_ratio_th = 0.1
enlarge_tests_num_by_factors = [ 0.7, 0.8, 0.9, 1]#[0.75, 0.9, 1, 1.25]# [0.5, 0.75, 1, 1.5]
Tbaseline           = 'ML' # options: 'ML', 'lb_no_priors', 'lb_with_priors'
methods_DD          = ['Normal']#{'Normal', 'Sum'} # options: Normal, Iterative, Sum
calc_Pu             = 1
third_step_type     = 'MLE' # options: ['scomp_dont_complete', 'scomp_complete', 'MAP', 'MLE']
calc_Pw             = 1
debug_mode          = False
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
        count_DND1 = np.zeros((numOfK, num_of_test_scale))
        count_PD1 = np.zeros((numOfK, num_of_test_scale))
        count_DD2 = np.zeros((numOfK, num_of_test_scale))
        count_DND3 = np.zeros((numOfK, num_of_test_scale))
        count_PD3 = np.zeros((numOfK, num_of_test_scale))
        count_unknown2 = np.zeros((numOfK, num_of_test_scale))
        count_unknown2_verif = np.zeros((numOfK, num_of_test_scale))
        count_success_DD = np.zeros((numOfK, num_of_test_scale))
        count_success_DD_non_exact = np.zeros((numOfK, num_of_test_scale))
        count_add_success_third_step = np.zeros((numOfK, num_of_test_scale))
        count_success_non_exact_third_step = np.zeros((numOfK, num_of_test_scale))
        count_not_detected = np.zeros((numOfK, num_of_test_scale))
        expected_notDetected = np.zeros((numOfK, num_of_test_scale))
        expected_DD = np.zeros((numOfK, num_of_test_scale))
        expected_PD = np.zeros((numOfK, num_of_test_scale))
        expected_unknown = np.zeros((numOfK, num_of_test_scale))
        iter_of_true_permute = np.zeros((numOfK, num_of_test_scale, nmc))
        Pw_of_true_out_of_max_Pw = np.zeros((numOfK, num_of_test_scale, nmc))
        correct_Pw = np.zeros((numOfK, num_of_test_scale, nmc))
        correctPw_outof_maxPw = np.zeros((numOfK, num_of_test_scale, nmc))

        #%% Start simulation
        for idxK in range(numOfK):
            K = vecK[idxK]
            print('K = ' + str(K))
            # For each K calculate number of test according the Tml and scale factor
            ge_model = None
            if sample_method == 'GE': # create the ge model 
                _, ge_model = sample_population_gilbert_elliot_channel2(N, K, ge_model)
            vecT = calculate_vecT_for_K(K, N, enlarge_tests_num_by_factors, Tbaseline=Tbaseline, Pe=Pe, 
                        sample_method=sample_method, ge_model=ge_model, Pu=None, coeff_mat=None)
            vecTs.append(vecT)
            for idxT in range(num_of_test_scale):
                T = np.int16(vecT[idxT])
                print('T', T)
                overflow_const = 1 #10^T

                p = np.log(2)/K#1-2**(-1/K) # options: 1/K, log(2)/K, 1-2**(-1/K)
                expected_PD[idxK, idxT] = K + (N-K) * (1-p*(1-p)**K)**T 
                
                # expected_DD[idxK, idxT] = 0.5*p*(1-p)**(expected_PD[idxK, idxT]-1) #old, wrong
                # try 2:
                # for tt in range(T):
                #     expected_DD_in_row = (expected_PD[idxK, idxT]-tt)*p*(1-p)**(expected_PD[idxK, idxT]-tt-1)
                #     expected_DD[idxK, idxT] += expected_DD_in_row
                nPD = expected_PD[idxK, idxT]
                # expected_DD[idxK, idxT]  = nPD*(1-p*(1-p)**(nPD-1))**T # like in multi_level_GT
                # p_defective = K/nPD # p_defective options1: P(Y=1)=1-(1-p)**K # p_defective option2: K/N # p_defective option3: 1
                # mycalc                
                expected_DD[idxK, idxT]  = K*(1-(1-p*(1-p)**(nPD-1))**T)#nPD*p_defective*(1-(1-p*(1-p)**(nPD-1))**T) # version1 - p_defective appears once
                # expected_DD[idxK, idxT]  = nPD*(1-(1-p_d efective*p*(1-p)**(nPD-1))**T) # version 2 - take in account p_defective for each row probability
                expected_notDetected[idxK, idxT] = K - expected_DD[idxK, idxT]
                
                for nn in range(nmc):
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
                        U, ge_model = sample_population_gilbert_elliot_channel2(N, K, ge_model)
                    true_defective_set = np.where(U == 1)[1]                    ## 1. Definitely Not Defective
                    # Encoder - bernoulli 
                    X =  np.multiply(np.random.uniform(0,1,(T, N)) < p,1) # iid testing matrix
                    if typical_codes:
                        non_typical_rows = np.arange(T)
                        while len(non_typical_rows)>0:
                            # randomize new rows and insert in X
                            num_of_non_typical_rows = len(non_typical_rows)
                            newRows = np.random.uniform(0,1,(num_of_non_typical_rows, N)) < p
                            X[non_typical_rows,:] = newRows
                            # find nontypical rows
                            number_of_ones_in_X = np.sum(X[non_typical_rows,:],1)
                            number_of_zeros_in_X = N-number_of_ones_in_X
                            non_typical_rows = np.where(abs(number_of_ones_in_X / N - p) > ones_zeros_ratio_th)[0]
                    
                    tested_mat = X*U
                    Y = np.sum(tested_mat, 1) > 0 
        #             # count difference between ones and zeros in codebook
        #             number_of_ones_in_X = sum(X,2)
        #             number_of_zeros_in_X = N-number_of_ones_in_X
        #             zeros_and_ones_diff = [zeros_and_ones_diff, number_of_zeros_in_X- number_of_ones_in_X]
                    # Decoder - CoMa
                    PD1 = np.arange(N)
                    DND1 = []
                    for ii in range(T):
                        if len(PD1) <= K:
                            break 
                        if Y[ii] == 0:
                            for jj in PD1:
                                if X[ii,jj] == 1: # definitely not defected
                                    PD1 = PD1[PD1 != jj]
                                    DND1 += [jj]

                    count_DND1[idxK, idxT] += len(DND1)
                    count_PD1[idxK, idxT] += len(PD1)
                    # PD1 = PD1.tolist()
                    if len(PD1) <= K: # all the PD are DD - all defective found
                        count_DD2[idxK, idxT] += len(PD1)
                        count_success_DD[idxK, idxT] += 1
                        count_success_DD_non_exact[idxK, idxT] += 1
                        continue
                    ## 2. WORKING Definite Defective
                    # steps 1&2
                    if method_DD == 'Normal':
                        DD2 = []

                        for ii in range(T):
                            if Y[ii] == 1 and np.sum(X[ii,PD1]) == 1: # only 1 item among the PD equals 1 and the rest equal 0
                                jj = np.where(X[ii,PD1] == 1)[0][0] # find the definite defective item index in PD1 array
                                defective = PD1[jj]
                                if defective not in DD2: # add jj only if jj is not already detected as DD
                                    DD2 += [defective]
                        count_DD2[idxK, idxT] += len(DD2)

                        if len(DD2) >= K: # all defective found
        #                     fprintf('All defective found\n')
                            count_success_DD[idxK, idxT] += 1
                            count_success_DD_non_exact[idxK, idxT] +=1
                            continue
                        really_defective = np.where(U==1)[1].tolist()
                        num_of_false_positive_in_DD2 = [e for e in DD2 if e not in really_defective]
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
                                    
                        count_DD2[idxK, idxT] += len(DD2)
                        if len(DD2) >= K: # all defective found
        #                     fprintf('All defective found\n')
                            count_success_DD[idxK, idxT] += 1
                            count_success_DD_non_exact[idxK, idxT] += 1
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
                                    
                        count_DD2[idxK, idxT] += len(DD2)
                        if len(DD2) >= K: # all defective found
        #                     fprintf('All defective found\n')
                            count_success_DD[idxK, idxT] += 1
                            count_success_DD_non_exact[idxK, idxT] += 1
                            continue 
        #                 if ~isempty(helpful_iter) && length(helpful_iter) > 1
        #                     fprintf([num2str(length(helpful_iter)) ' helpful iter\n'])

                    # find all unknown
                    unknown2 = [e for e in PD1 if e not in DD2]#PD1[PD1 not in DD2][0]
                    # print('#PD = {} || #DD = {} || #unknown = {}'.format(len(PD1), len(DD2), len(unknown2)))
                    count_unknown2[idxK, idxT] += len(unknown2)
                    count_unknown2_verif[idxK, idxT] += len(PD1) - len(DD2)
                    count_not_detected_defectives = K-len(DD2)
                    count_success_DD_non_exact[idxK, idxT] += (len(DD2) / K)

                    ## 3. MAP?
                    if not do_third_step:
                        continue

                    if third_step_type == 'MAP':
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
                        # item detected as defective gets prob 1 to be defective in this step:
                        idx_of_DD_in_PD_coor = [list(PD1).index(item) for item in DD2]
                        # Pu_DD2[0,idx_of_DD_in_PD_coor] = 1
                        # 3.2 start MAP
                        # print('#unknown = {}, #not_detected_yet = {}'.format(len(unknown2), count_not_detected_defectives))
                        # if len(unknown2) - count_not_detected_defectives > 30:
                        #     continue
                        
                        if sample_method == 'GE':

                            if add_dd_based_prior: 
                                pass

                            if is_sort_comb_by_priors:
                                # print('sort permutations')
                                # all_permutations3, Pw_sorted, Pu3 = ge_model.sort_comb_by_priors_GE(N, all_permutations3, DD2, DND1)
                                all_permutations3, Pw_sorted = ge_model.sort_comb_by_priors_GE_cut_by_entropy(K, T, nPD, DD2, DND1, unknown2)

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
                            Pu3[0,idx_of_DD_in_PD_coor] = 1                        
                        
                            if is_sort_comb_by_priors:
                                Pu3_all_population = np.zeros((1, N))
                                Pu3_all_population[0,PD1] = Pu3
                                Pu3 = Pu3_all_population
                                Pu3 = Pu3 @ coeff_mat # not only initial prbabilitis but 
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
                                if debug_mode and set(permute) == set(true_defective_set):
                                    print('Yw≠Y')
                                continue
                            valid_options += 1
                            apriori[comb] = 1e16#1e32 #overflow_const
                            if sample_method == 'ISI':
                                apriori[comb] *= Pw_sorted[comb] * p ** np.sum(X_forW == 1)
                            elif sample_method == 'GE':
                                # calc P(w|Xsw)
                                Pw_by_Xsw = 1e16
                                for tt in range(T):
                                    participating_items = np.where(X_forW[tt,:] == 1)[0]
                                    if participating_items.shape[0] == 0:
                                        continue

                                    probability_per_item = [ge_model.get_conditional_probability_GE(item, DD2, DND1) for item in participating_items]
                                    Pw_by_Xsw *= np.sum(probability_per_item)
                                
                                apriori[comb] = Pw_by_Xsw * p ** np.sum(X_forW == 1)
                                pass

                            else:
                                for tt in range(T):
                                    num_of_ones_in_Xwt = np.sum(X_forW[tt,:] == 1)
                                    P_q = overflow_const * p**num_of_ones_in_Xwt
                                    P_Xsw_t = P_q
                                    for ii in permute: # TODO: do I need to add DD2?
                                        P_Xsw_t *= Pu3[0,ii]
                                    apriori[comb] *= P_Xsw_t # get zeros
                            
                            if set(permute+DD2) == set(true_defective_set) and debug_mode:
                                print('true defective set prior: prior(W*) = ' + str(apriori[comb,0]))

                        max_likelihood_W = np.argmax(apriori)
                        if debug_mode:
                            print('chosen defective set prior: prior(estW) = ' + str(apriori[max_likelihood_W,0]))
                        estU = np.zeros(U.shape)
                        estU[0, all_permutations3[max_likelihood_W,:]] = 1  
                        estU[0, DD2] = 1
                        iter_of_true_permute[idxK, idxT, nn] = max_likelihood_W
                        Pw_of_true_out_of_max_Pw[idxK, idxT, nn] = Pw_sorted[max_likelihood_W] / Pw_sorted[0]

                        ''' 
                        TODO: I can decrease the num_of_permutations when 
                        nchoosek (unknown, not detected)<<#permutations I find using 2^(T*error_entropy)
                        '''
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
                            # permute = all_permutations3_no_prior[w,:]
                            U_forW = np.zeros((1,N))
                            U_forW[0,permute] = 1
                            X_forW = X*U_forW
                            Y_forW = np.sum(X_forW, 1) > 0
                            # evaluate Y
                            # mle_error_counter[0, w] = sum(Y_forW != Y)
                            # if mle_error_counter[0, w] == 0:
                            #     max_likelihood_W = w
                            #     break
                            error_counter = np.sum(Y_forW != Y)
                            if error_counter < min_error_counter:
                               min_error_counter = error_counter
                               max_likelihood_W = w
                               if error_counter == 0: # Yw = Y
                                   break 
                        # if max_likelihood_W is None:
                        #     max_likelihood_W = np.argmin(mle_error_counter)
                        estU = np.zeros(U.shape)
                        estU[0, all_permutations3_no_prior[max_likelihood_W,:]] = 1  
                        estU[0, DD2] = 1

                    elif third_step_type == 'scomp_dont_complete':
                        # (1) try the first subset
                        try_subset = DD2
                        try_U = np.zeros(U.shape)
                        while True:#not satisfying:
                            # (2.1) try the current defective subset
                            try_U = np.zeros(U.shape)
                            try_U[0,try_subset] = 1
                            try_tested_mat = X*try_U
                            try_Y = np.sum(try_tested_mat, 1) > 0
                            # check if the set is satisfying
                            satisfying = (Y == try_Y).all() 
                            if satisfying or len(try_subset) == K:
                                break

                            # (2.2) find the element which appears in the largest number of tests which are unexplained by K
                            count_times_participating = np.zeros((len(unknown2),))
                            unexplained_tests = np.where(Y != try_Y)[0]
                            for unexplained_test in unexplained_tests:
                                unknown_participants = np.where(X[unexplained_test, unknown2] == 1)[0]
                                count_times_participating[unknown_participants] += 1
                                 
                            max_participating = unknown2[np.argmax(count_times_participating)] # index in 1,..,N coordinates
                            try_subset.append(max_participating) # add new item to the subset 

                        pass

                        estU = try_U
                    if np.sum(U != estU) == 0:
                        count_add_success_third_step[idxK, idxT] += 1
                        count_success_non_exact_third_step[idxK, idxT] += (K-len(DD2))/K
                    else:
                        # count only the items detected in the 3rd step 
                        detected_defectives = np.where(estU==1)[1] # may be errornous detection
                        not_detected = set(true_defective_set)-set(detected_defectives)
                        num_of_correct_detection = K-len(not_detected)
                        count_success_non_exact_third_step[idxK, idxT] += (num_of_correct_detection-len(DD2))/K 
                        
        # Normalize success and counters
        
        count_success_DD *= 100/nmc
        count_add_success_third_step *= 100/nmc
        count_success_exact_tot = count_success_DD + count_add_success_third_step
        count_success_DD_non_exact *= 100/nmc
        count_success_non_exact_third_step *= 100/nmc
        count_success_non_exact_tot = count_success_DD_non_exact + count_success_non_exact_third_step
        print('count_success_exact_tot', count_success_exact_tot)
        print('count_success_exact_non_exact_tot', count_success_non_exact_tot)
        
        count_DND1 = count_DND1 / nmc
        count_PD1 = count_PD1 / nmc
        count_DD2 = count_DD2 / nmc
        count_DND3 = count_DND3 / nmc
        count_PD3 = count_PD3 / nmc
        count_unknown2 = count_unknown2 / (nmc - count_success_DD) 
        count_unknown2_verif = count_unknown2_verif / (nmc - count_success_DD) 
        expected_unknown = expected_PD - expected_DD
        count_not_detected = np.matlib.repmat(np.array(vecK), num_of_test_scale,1).T - count_DD2
        
        # Make resutls directory
        results_dir_path = None

        typical_label = '_nottypical'
        if typical_codes:
            typical_label = '_typical'
        
        third_step_label = third_step_type
        if not do_third_step:
            third_step = 'None'

        if save_fig or save_raw:    
            time_str = datetime.now().strftime("%d%m%Y_%H%M%S")
            experiment_str = 'N' + str(N) + '_nmc' + str(nmc) + '_methodDD_' + method_DD + '_thirdStep_' + third_step_label + typical_label + '_Tbaseline_' +  Tbaseline + '_'
            results_dir_path = os.path.join(save_path, 'countPDandDD_' + experiment_str + time_str)
            os.mkdir(results_dir_path)
        
        #%% Visualize
        if is_plot:
            plot_DD_vs_K_and_T(N, vecT, vecK, count_PD1, enlarge_tests_num_by_factors, nmc, count_DD2, sample_method, 
                                method_DD, Tbaseline, typical_codes, results_dir_path)
            plot_expected_DD(vecK, expected_DD, count_DD2, vecT, enlarge_tests_num_by_factors, results_dir_path)
            plot_expected_PD(vecK, expected_PD, count_PD1, vecT, enlarge_tests_num_by_factors, results_dir_path)
            plot_expected_unknown(vecK, expected_unknown, count_unknown2, vecT, enlarge_tests_num_by_factors, results_dir_path)
            plot_expected_not_detected(vecK, expected_notDetected, count_not_detected, vecT, enlarge_tests_num_by_factors, results_dir_path)
            plot_expected_unknown_avg(vecK, expected_unknown, count_PD1 - count_DD2, vecT, 
                                    enlarge_tests_num_by_factors, results_dir_path)
            plot_Psuccess_vs_T(vecTs, count_success_DD, count_success_exact_tot, vecK, N, nmc, third_step_type, sample_method, 
                                method_DD, Tbaseline, enlarge_tests_num_by_factors, typical_label,
                                results_dir_path, exact=True)
            plot_Psuccess_vs_T(vecTs, count_success_DD_non_exact, count_success_non_exact_tot, vecK, N, nmc, third_step_type, sample_method, 
                                method_DD, Tbaseline, enlarge_tests_num_by_factors, typical_label,
                                results_dir_path, exact=False)

        #%% Save
        if save_raw:
            fullRawPath = os.path.join(results_dir_path, 'workspace.mat')
            all_variables_names = dir()
            # remove packages(numpy), functions(calculatePu), set type('not_detected'),... 
            dont_include_variables = ['np', 'numpy', 'pd', 'math', 'itertools', 'random', 'go', 'px', 'datetime', 'os', 'plt', 'shelve', 'reverse', \
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

            
