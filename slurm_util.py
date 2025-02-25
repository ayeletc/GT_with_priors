import os
import subprocess
import getpass
import time

def wait_for_available_job(max_jobs: int, sleep_time=10):
    """
    This function waits until a SLURM job can be submitted.
    It waits sleep_time seconds before trying to submit again.
    """
    while get_running_jobs() >= max_jobs:
        time.sleep(sleep_time)

def get_running_jobs():
    """
    This function gets the current number of running jobs.
    """
    user = getpass.getuser()
    result = subprocess.run(['squeue', '-u', user], stdout=subprocess.PIPE)
    return len(result.stdout.decode('utf-8').strip().split('\n')) - 1

def set_to_str(S: frozenset) -> str:
    """
    This function generates a string that looks like f"S_{s1}_{s2}_"...
    """
    S = sorted(S)
    return "S_" + "_".join(map(str, S))

def S_str_to_fname(S: frozenset, temp_res_dir: str) -> str:
    """
    This function returns the file name to store the results of the SLURM script that calculates calc_permutation_prob_standalone.
    """
    S_str = set_to_str(S)
    return os.path.join(temp_res_dir, f"{S_str}.txt")

def calc_perm_slurm(S: frozenset, temp_res_dir: str, N: int, s: float, q: float, pi_B: float, qos: str, time_min: int = 10):
    """
    This function runs a SLURM job that will calculate calc_permutation_prob_standalone.
    It returns the path to the SLURM script.
    """
    S_str = set_to_str(S)
    fname_out = "/dev/null"
    fname_err = "/dev/null"
    # fname_out = os.path.join(temp_res_dir, f"{S_str}.out")
    # fname_err = os.path.join(temp_res_dir, f"{S_str}.err")
    fname_res = S_str_to_fname(S, temp_res_dir)

    # #DEBUG ONLY
    # from calc_permutation_prob_standalone import calc_permutation_prob_standalone
    # calc_permutation_prob_standalone(S_str, N, s, q, pi_B, fname_res)


    command = f'''sbatch --parsable --job-name="{S_str}" --qos="{qos}" --time={time_min} --nodes=1 --ntasks=1 --wrap="python -m calc_permutation_prob_standalone --S_str {S_str} --N {N} --q {q} --s {s} --pi_B {pi_B} --fname_res {fname_res}" --output {fname_out} --error {fname_err}'''
    std = subprocess.run(command, shell=True, capture_output=True)
    if std.stderr.decode('UTF-8'):
        print(f"Warning! Error {std.stderr.decode('UTF-8')} in S_str={S_str}.")
    # print("STDOUT:", std.stdout.decode('utf-8'))
    # print("STDERR:", std.stderr.decode('utf-8'))
