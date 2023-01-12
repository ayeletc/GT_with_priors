import math
import random
import numpy as np
from utils import *


class GE_model:
    def __init__(self, s, q, pi_B):
        self.s = s
        self.q = q
        self.pi_B = pi_B
        self.probabilities_to_bad_dict = self.calc_conditional_probability_GE() 
        self.num_of_permutations = None
    
    def sample_gilbert_elliot_channel(self, N, max_bad=np.inf):
        # for GT with fixed num of K : if there are more than max_bad bad items, return false and don't complete the chain

        # P = np.array([[1-self.q, self.q], [self.s, 1-self.s]])
        # pi_G = self.s/(self.s+self.q) # probability of being in State G
        # pi_B = self.q/(self.s+self.q) # probability of being in State B
    
        channel_statef = np.zeros((N,))
        channel_stater = np.zeros((N,))
        
        goodf = random.random() > self.pi_B
        goodr = random.random() > self.pi_B          
        
        num_of_bad = 0
        for ii in range(N):
            # set goodf = 1 and goodr = 1 if next step is bad (=erasure/defective)
            if goodf == 1 and goodr == 1:
                goodf = random.random() > self.q  
                goodr = random.random() > self.q  
            elif goodf == 1 and goodr == 0:
                goodf = random.random() > self.q  
                goodr = random.random() > 1-self.s
            elif goodf == 0 and goodr == 1:
                goodf = random.random() > 1-self.s 
                goodr = random.random() > self.q 
            elif goodf == 0 and goodr == 0:
                goodf = random.random() > 1-self.s 
                goodr = random.random() > 1-self.s

            channel_statef[ii] = goodf
            channel_stater[ii] = goodr
            if goodf == 0:
                num_of_bad += 1
                if num_of_bad > max_bad:
                    return None, None
        return 1-channel_statef,1-channel_stater


    def calc_conditional_probability_GE(self):
        probabilities_to_bad_dict = {}
        # given previous item is defective
        probabilities_to_bad_dict['previous_is_defective'] = 1-self.s
        # given previous item is not defective
        probabilities_to_bad_dict['previous_is_not_defective'] = self.q
        # no given prior about previous item
        probabilities_to_bad_dict['first_item'] = self.pi_B
        probabilities_to_bad_dict['no_prior_given'] = self.pi_B# * (1-self.s) + (1-self.pi_B) * self.q
        return probabilities_to_bad_dict

    def get_conditional_probability_GE(self, item, DD2, DND1):
        if item-1 in DD2:
            # given previous item is defective
            return self.probabilities_to_bad_dict['previous_is_defective']
        elif item-1 in DND1:
            # given previous item is not defective
            return self.probabilities_to_bad_dict['previous_is_not_defective']
        else:
            # no given prior about previous item
            if item == 0: # first item, the probabiity is not conditional
                return self.probabilities_to_bad_dict['first_item']
            else:
                return self.probabilities_to_bad_dict['no_prior_given']

    def calculate_lower_bound_GE(self, N, Pe=0.0):
        return self.calculate_entropy(N) * (1-Pe)
    
    def calculate_entropy(self, N):
        return N * ( (1-self.pi_B) * (-self.q*np.log2(self.q)-(1-self.q)*np.log2(1-self.q)) +   \
                    self.pi_B * (-self.s*np.log2(self.s)-(1-self.s)*np.log2(1-self.s)) )
    '''
    def sort_comb_by_priors_GE(self, N, all_permutations, DD2, DND1):
        Pw = np.ones((all_permutations.shape[0],))
        Pu_with_priors = np.zeros((1,N))
        \''' 
        calculate Pw
        P(W) = P(U) = P(Ud1) * P(Ud2|Ud1) * .... * P(Udk|Udk-1, ..., Ud1) 
        In GE P(Udk|Udk-1, ..., Ud1) = P(Udk|Udk-1)  [markov chain]
        \'''
        for ii, permute in enumerate(all_permutations):
            permute = sorted(permute.tolist() + DD2)
            if permute[0] in DD2:
                Pw[ii] = 1
            elif permute[0] in DND1:
                Pw[ii] = 0
                continue
            else:
                Pw[ii] = self.pi_B

            for jj in range(1,len(permute)):
                p_item_is_defective_given_previous = self.get_conditional_probability_GE(permute[jj-1], DD2, DND1)
                Pw[ii] *= p_item_is_defective_given_previous
            
            Pu_with_priors[0,permute] += Pw[ii] 
        # sort all permutations by their probabilities Pw
        Pw_idx = Pw.argsort()[::-1] # descending order, first has the highest probability
        Pw_sorted = Pw[Pw_idx]
        if Pw_sorted[0] == 0:
            print('overflow?')
            pass
        all_permutations_sorted = all_permutations[Pw_idx,:]
        # normalize Pu_with_priors
        Pu_with_priors /= np.sum(Pu_with_priors)
        return all_permutations_sorted, Pw_sorted, Pu_with_priors
    '''
    def sort_comb_by_priors_GE_cut_by_entropy(self, N, K, T, nPD, DD2, DND1, unknowns, permutation_factor=50):
        '''
        1. calculate Np the number of permutations we want to check - 2^( T*H(Perror_dd) )
        1.1. calculate Perror_dd

        2. define array of probabilities of permutations Pw 
            and array Mp (Np x K_left) of the permutation with the highest Pw
        
        3. calculate the permutations one by one, for each one:
        3.1. calculate Pw
        3.2 if this Pw high enough put it in the array Mp of the permutations (keep both arrays sorted)
        '''
        K_left = K - len(DD2)
        # if K_left == 1:
        #     permute =
        #     return Pw, high_prob_permutations
        ''' 
        Perror in DD:
        (1-p) + p*(Phidden)
        Phidden = 1-P(there are no more PDs on) = 1-P( (#PD-1) items are off ) = 1-(1-p)^(#PD-1)
        '''
        if self.num_of_permutations is None:
            p = np.log(2) / K
            prob_error_DD = 1-p*(1-p)**(nPD-1)
            entropy_error_DD = -prob_error_DD * np.log2(prob_error_DD) - (1-prob_error_DD) * np.log2(1-prob_error_DD)
            self.num_of_permutations = np.ceil(2 ** (T * entropy_error_DD)).astype(np.int64)

        # print('self.num_of_permutations', self.num_of_permutations)
        
        num_of_permutations_binomial = math.comb(len(unknowns), K_left)
        # print('num_of_permutations_binomial', num_of_permutations_binomial)
        # if num_of_permutations_binomial < 500 and num_of_permutations_binomial > self.num_of_permutations:
        #     save_permutations = num_of_permutations_binomial
        # else:
        #     save_permutations = np.min([self.num_of_permutations, num_of_permutations_binomial])
        if num_of_permutations_binomial > self.num_of_permutations:
            save_permutations = permutation_factor*self.num_of_permutations
        else:
            save_permutations = self.num_of_permutations
        Pw = np.zeros((save_permutations,))
        high_prob_permutations = np.zeros((save_permutations, K_left))
        num_of_iterations_in_sort = num_of_permutations_binomial
        # if num_of_permutations_binomial < self.num_of_permutations:
        #     # built the iterator on all the possible options and sort
        # else:
        # calculate permutations one by one:
        iterable = unknowns
        r = K_left
        pool = tuple(iterable)
        n = len(pool)
        r = n if r is None else r
        if r > n:
            print('Do something1')
            return
        indices = list(range(r))
        comb = tuple(pool[i] for i in indices)
        # print(comb)
        prob_permute = self.calc_Pw_fixed(N, comb, DD2, DND1)
        Pw, high_prob_permutations = add_new_value_and_symbol_keep_sort(Pw, high_prob_permutations, prob_permute, comb)
        # iteration = 1
        while True:
            for i in reversed(range(r)):
                if indices[i] != i + n - r:
                    break
            else:
                # print('#iterations in sort = ', iteration)
                return high_prob_permutations.astype(np.int64), Pw, num_of_iterations_in_sort
            indices[i] += 1
            for j in range(i+1, r):
                indices[j] = indices[j-1] + 1
            comb = tuple(pool[i] for i in indices)
            # print(comb)
            prob_permute = self.calc_Pw_fixed(N, comb, DD2, DND1)
            Pw, high_prob_permutations = add_new_value_and_symbol_keep_sort(Pw, high_prob_permutations, prob_permute, comb)
            # iteration += 1        

    def calc_Pw(self, permute, DD2, DND1):        
        # probability of the first item in the permutation:
        if permute[0] in DD2:
            Pw = 1
        elif permute[0] in DND1:
            Pw = 0
        else:
            Pw = self.pi_B
        # multiply transition probabilities
        for jj in range(1,len(permute)):
            p_item_is_defective_given_previous = self.get_conditional_probability_GE(permute[jj-1], DD2, DND1)
            Pw *= p_item_is_defective_given_previous
        return Pw

    def calc_Pw_fixed(self, N, permute, DD2, DND1):  # take in account all n items
        # probability of the first item in the permutation:
        first_item = 0
        if first_item in DD2:
            Pw = 1
        elif first_item in DND1:
            Pw = 0
        else:
            Pw = self.pi_B
        # multiply transition probabilities
        for jj in range(1,N):
            if jj in permute:
                p_item_is_defective_given_previous = self.get_conditional_probability_GE(jj, DD2, DND1)
            else:
                p_item_is_defective_given_previous = 1-self.get_conditional_probability_GE(jj, DD2, DND1)
            Pw *= p_item_is_defective_given_previous
        return Pw
