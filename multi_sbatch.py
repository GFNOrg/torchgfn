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
    return f"sbatch --time {args.time} {gres} --job-name {job_name} -c 4 --mem {args.mem} {suffix} run.sh "


no_cuda = "--no_cuda"
wandb = "TB_vs_HVI_schedule"
validation_samples = 200000
counter = 0
for seed in (11, 12, 13, 14):
    for (ndim, height) in [(2, 64), (4, 8)]:
        for (use_tb, use_baseline, v2, use_chi2) in [
            # ("--use_tb", "", "", ""),
            # ("", "--use_baseline", "", ""),
            ("", "", "--v2", "--use_chi2"),
            ("", "--use_baseline", "--v2", "--use_chi2"),
        ]:
            job_name = f"{counter}_{wandb}"
            script_to_run = f"""python compare_TB_to_VI.py --ndim {ndim} --height {height} 
                                --preprocessor KHot --batch_size 16 --n_iterations 100000 
                                --lr 0.001 --lr_Z 0.1 --learn_PB --tie_PB {no_cuda}
                                {use_tb} {use_baseline} --wandb {wandb} --seed {seed}  {use_chi2} {v2} 
                                --validation_samples {validation_samples} --validation_interval 100 
                                --validate_with_training_examples"""
            script_to_run = script_to_run.replace("\n", " ")
            print(f"{counter}, {script_to_run}")
            subprocess.check_output(prefix(job_name) + script_to_run, shell=True)
            counter += 1
