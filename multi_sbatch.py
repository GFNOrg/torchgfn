import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--no-cuda", action="store_true", default=False, help="use gpu")
parser.add_argument("--time", default="2:00:0", help="time")
parser.add_argument("--mem", default="12G", help="memory needed for each job")
parser.add_argument("--account", default="", type=str, help="account for slurm sbatch")
args = parser.parse_args()

gres = "--gres=gpu:1" if not args.no_cuda else ""
suffix = f"--account={args.account}" if args.account else ""


def prefix(job_name):
    return f"sbatch --time {args.time} {gres} --job-name {job_name} -c 1 --mem {args.mem} {suffix} run.sh "


job_name = "0_reverse_kl"
scripts = "python four_kls.py --mode rws         --n_iterations 40000 --batch_size 64  --sample_from_reward            --baseline None   --seed 1337 --wandb gfn_kls_fix1 &> /dev/null & python four_kls.py --mode rws         --n_iterations 40000 --batch_size 64                       --reweight --baseline None   --seed 1337 --wandb gfn_kls_fix1 &> /dev/null & python four_kls.py --mode rws         --n_iterations 40000 --batch_size 64  --sample_from_reward --reweight --baseline None   --seed 1337 --wandb gfn_kls_fix1 &> /dev/null & python four_kls.py --mode rws         --n_iterations 40000 --batch_size 64                                  --baseline None   --seed 1337 --wandb gfn_kls_fix1 &> /dev/null &"
subprocess.check_output(prefix(job_name) + scripts, shell=True)
