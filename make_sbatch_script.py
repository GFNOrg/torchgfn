import os
import subprocess
import stat
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--job_name", type=str, default=None)
parser.add_argument("--n_threads_per_task", type=int, default=10)
parser.add_argument("--ntasks_per_node", type=int, default=6)
parser.add_argument("--partition", type=str, default="main")
parser.add_argument("--offset", type=int, default=0)

args = parser.parse_args()

job_name = args.job_name if args.job_name is not None else f"four_kls_{args.offset}"
output_filename = f"/network/scratch/l/lahlosal/SLURM_OUTPUTS/{job_name}"
ntasks_per_node = args.ntasks_per_node
partition = args.partition
gres = "gpu:1"
cpus_per_task = 1
mem = "10G"

conda_env = "gfn"

wandb_dir = "/home/mila/l/lahlosal/scratch/wandb"
models_directory = "/home/mila/l/lahlosal/scratch/four_kls_models"


sbatch_directory = "sbatch_scripts"
if not os.path.exists(sbatch_directory):
    os.makedirs(sbatch_directory)

bash_range = "{1.." + str(args.n_threads_per_task) + "}"
sbatch_skeleton = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_filename}.out
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}

module load anaconda/3
conda activate {conda_env}


srun --output={output_filename}-%t.out bash -c 'for i in {bash_range}; do python four_kls.py --task_id=$i --total={args.n_threads_per_task} --offset={args.offset} --wandb_dir={wandb_dir} --models_directory={models_directory}& done; wait;'
"""


with open(f"{sbatch_directory}/{job_name}.sh", "w+") as f:
    f.writelines(sbatch_skeleton)

st = os.stat(f"{sbatch_directory}/{job_name}.sh")
os.chmod(f"{sbatch_directory}/{job_name}.sh", st.st_mode | stat.S_IEXEC)

subprocess.check_output(f"sbatch {sbatch_directory}/{job_name}.sh", shell=True)
