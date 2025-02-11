import unittest
from itertools import chain, combinations

from GE_model import GE_model
from utils import all_subsets

class TestGETotalProbabilities(unittest.TestCase):
    def setUp(self):
        self.q = 0.2
        self.s = 0.3
        self.pi_B = 0.4
        self.ge = GE_model(self.s, self.q, self.pi_B)
    
    def _test_prob(self, K: int, N: int, expected: float) -> float:
        prob = self.ge.calc_total_ones_prob(K=K,N=N)
        self.assertAlmostEqual(expected, prob)

    def test_N_1(self):
        K=1
        N=1

        expected = self.pi_B
        self._test_prob(K=K, N=N, expected=expected)

    def test_N_2(self):
        K = 1
        N = 2

        expected = self.pi_B*self.s + (1-self.pi_B)*self.q
        self._test_prob(K=K, N=N, expected=expected)

class TestGEIndividualProbabilities(unittest.TestCase):
    def setUp(self):
        self.q = 0.2
        self.s = 0.3
        self.pi_B = 0.4
        self.ge = GE_model(self.s, self.q, self.pi_B)

    def test_K_1(self):
        N = 10
        
        #Analytical calculation
        pr_start_1 = self.pi_B*self.s * ((1-self.q)**(N-2))
        pr_end_1 = (1-self.pi_B) * ((1-self.q)**(N-2)) * self.q
        pr_others = (N-2)*((1-self.pi_B) * self.q * ((1-self.q)**(N-3)) * self.s)
        expected =  pr_start_1+pr_others+pr_end_1
        
        #Code calculation
        calculated = sum([self.ge.calc_permutation_prob(defectives=[curr_set], N=N) for curr_set in range(N)])
        self.assertAlmostEqual(expected, calculated)

    def test_sum_to_one(self):
        N = 4
        all_sets = all_subsets(N)
        res = sum([self.ge.calc_permutation_prob(defectives=curr_set, N=N) for curr_set in all_sets])
        self.assertAlmostEqual(res, 1.0)
