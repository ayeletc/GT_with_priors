import unittest

from GE_model import GE_model

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
        K = 1
        N = 10
        #TODO: Finish this test.
