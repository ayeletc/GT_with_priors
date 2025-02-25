import argparse
import os

from utils import ExecutionMode
from GE_model import GE_model

def calc_permutation_prob_standalone(S_str: str, N: int, s: float, q: float, pi_B: float, fname_res: str) -> float:
    """
    This function calculates GE_model.calc_permutation_prob but without the class.
    It is useful for SLURM scripts, as a GE_model is not passed but created on the fly.
    """
    S = frozenset([int(S) for S in S_str.split("_")[1:]]) #The first element is "S" and is not needed.
    ge_model = GE_model(s=s, q=q, pi_B=pi_B)
    #From the perspective of the call, this is a single call hence sequential and NOT parallel_slurm.
    result = ge_model.calc_permutation_prob(defectives=S, N=N, exec_mode=ExecutionMode.SEQUENTIAL, temp_res_dir=os.path.dirname(fname_res))
    with open(fname_res, "w") as f:
        f.write(str(result))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--S_str", type=str, help="Defective items, separated by '_'.")
    parser.add_argument("--N", type=int, help="Number of items.")
    parser.add_argument("--q", type=float, help="probability to move from 0 state to 1 state. If -1, will be calculated on the fly by the code.", default=-1.)
    parser.add_argument("--s", type=float, help="Probability to move from 1 state to 0 state. If -1, will be calculated on the fly by the code.", default=-1.)
    parser.add_argument("--pi_B", type=float, help="Probability to start at state 1.")
    parser.add_argument("--fname_res", type=str, help="Where to write the result.")
    args = parser.parse_args()

    S = args.S_str
    N = args.N
    q = args.q
    s = args.s
    pi_B = args.pi_B
    fname_res = args.fname_res
    calc_permutation_prob_standalone(S_str=S, N=N, s=s, q=q, pi_B=pi_B, fname_res=fname_res)