import argparse
import time

from sample_population import sample_population_gilbert_elliot_channel

if __name__=="__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, help="Number of defective items.")
    parser.add_argument("--N", type=int, help="Number of items.")
    parser.add_argument("--eps", type=float, help="Epsilon for the bound.", default=0.)
    parser.add_argument("--parallel", dest="parallel", action="store_true", help="If True, train the clients in parallel.")
    parser.add_argument("--debug", dest="debug", action="store_true", help="If True, run in debug mode.")
    args = parser.parse_args()

    K = args.K
    N = args.N
    eps = args.eps
    parallel = args.parallel
    debug = args.debug

    print(f"K={K}, N={N}, eps={eps}")
    _, ge_model = sample_population_gilbert_elliot_channel(N, K, None, debug=debug)
    bound = max([(1+eps)*(K/i)*ge_model.calc_entropy_s2_given_s1(K=K, N=N, i=i) for i in range(1,K+1)])
    end_time = time.time()
    elapsed = end_time-start_time
    print(f"The bound (without epsilon) is {bound}.")
    print(f"It took {elapsed:.3f} seconds (or {elapsed/3600:.3f} hours) to calculate.")