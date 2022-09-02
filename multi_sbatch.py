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


no_cuda = "--no_cuda"
wandb = "TB_vs_HVI_schedule_R0_2"
validation_samples = 200000
counter = 0
for seed in (21, 22, 23):
    for (ndim, height) in [(2, 64), (4, 8)]:
        for R0 in [0.1, 0.01, 0.001]:
            for (use_tb, use_baseline) in [
                ("--use_tb", ""),
                ("", "--use_baseline"),
            ]:
                for lr in [0.001, 0.0005, 0.0001]:
                    for lr_Z in [0.1, 0.01, 0.05]:
                        for schedule in (1.0, 0.99, 0.95, 0.9):
                            job_name = f"{counter}_{wandb}"
                            script_to_run = f"""python compare_TB_to_VI.py --ndim {ndim} --height {height}  --R0 {R0}
                                                --preprocessor KHot --batch_size 16 --n_iterations 100000 
                                                --lr {lr} --lr_Z {lr_Z} --learn_PB --tie_PB {no_cuda} --schedule {schedule}
                                                {use_tb} {use_baseline} --wandb {wandb} --seed {seed}
                                                --validation_samples {validation_samples} --validation_interval 100 
                                                --validate_with_training_examples"""
                            script_to_run = script_to_run.replace("\n", " ").replace(
                                "\t", " "
                            )
                            print(f"{counter}, {script_to_run}")
                            subprocess.check_output(
                                prefix(job_name) + script_to_run, shell=True
                            )
                            counter += 1
