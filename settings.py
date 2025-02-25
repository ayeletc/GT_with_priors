from enum import Enum

class STATE(Enum):
    CURR_ZERO = 0
    CURR_ONE = 1
    CURR_NONE = 2 #Used for the first instance of the chain

MAX_SLURM_JOBS = 2000 #Maximum allowed of SLURM jobs to be run at a time.