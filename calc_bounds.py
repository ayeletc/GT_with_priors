import argparse
import os
import time
import numpy as np
from utils import convert_to_exec_mode

from sample_population import sample_population_gilbert_elliot_channel

if __name__=="__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, help="Number of defective items.")
    parser.add_argument("--N", type=int, help="Number of items.")
    parser.add_argument("--eps", type=float, help="Epsilon for the bound.", default=0.)
    parser.add_argument("--q", type=float, help="probability to move from 0 state to 1 state. If -1, will be calculated on the fly by the code.", default=-1.)
    parser.add_argument("--s", type=float, help="Probability to move from 1 state to 0 state. If -1, will be calculated on the fly by the code.", default=-1.)
    parser.add_argument("--debug", dest="debug", action="store_true", help="If True, run in debug mode.")
    parser.add_argument("--exec_mode", type=str, default="sequential", help="Execution mode. Options: 'sequential', 'parallel_joblib', 'parallel_slurm'. Default: 'sequential'.")
    parser.add_argument("--temp_res_dir", type=str, help="Where to save temporary SLURM files. Must be passed if running in SLURM mode.", default="")
    parser.add_argument("--qos", type=str, help="QoS for SLURM scripts.", default="")
    args = parser.parse_args()

    K = args.K
    N = args.N
    eps = args.eps
    q = args.q
    s = args.s
    debug = args.debug
    exec_mode = convert_to_exec_mode(args.exec_mode)
    temp_res_dir = args.temp_res_dir
    qos = args.qos

    if temp_res_dir:
        os.makedirs(temp_res_dir, exist_ok=True)
    _, ge_model = sample_population_gilbert_elliot_channel(N, K, None, debug=debug)
    if q>=0:
        ge_model.q = q
    if s>=0:
        ge_model.s = s
    pi_B = ge_model.q/(ge_model.q+ge_model.s)
    ge_model.pi_B = pi_B
    print(f"K={K}, N={N}, eps={eps}, q={q}, s={s}, pi={pi_B}")
    print(f"Converse bound (with Pe=0) is {ge_model.calculate_lower_bound_GE(N)}.")
    bound = max([(1+eps)*(K/i)*ge_model.calc_entropy_s2_given_s1(K=K, N=N, i=i, exec_mode=exec_mode, temp_res_dir=temp_res_dir, qos=qos) for i in range(1,K+1)])
    end_time = time.time()
    elapsed = end_time-start_time
    print(f"The bound (without epsilon) is {bound}.")
    print(f"It took {elapsed:.3f} seconds (or {elapsed/3600:.3f} hours) to calculate.")
