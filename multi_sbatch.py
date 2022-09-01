import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--no-cuda", action="store_true", default=False, help="use gpu")
parser.add_argument("--time", default="2:00:0", help="time")
parser.add_argument("--mem", default="12G", help="memory needed for each job")
parser.add_argument("--suffix", default="", help="suffix for slurm sbatch")
args = parser.parse_args()

gres = "--gres=gpu:1" if not args.no_cuda else ""


def prefix(job_name):
    return f"sbatch --time {args.time} {gres} --job-name {job_name} -c 4 --mem {args.mem} {args.suffix} run.sh "


no_cuda = "--no_cuda"
wandb = "TB_vs_VI_tabular_v2"
validation_samples = 200000
counter = 0
for seed in (1, 2):
    for (ndim, height) in [(2, 64), (4, 8)]:
        for (use_tb, v2, use_baseline) in [
            ("--use_tb", "", ""),
            ("", "", "--use_baseline"),
            ("", "--v2", "--use_baseline"),
        ]:
            for (learn_PB, tie_PB) in [
                ("--learn_PB", "--tie_PB"),
                ("", ""),
            ]:
                job_name = f"{counter}_{wandb}"
                script_to_run = f"""python compare_TB_to_VI.py --ndim {ndim} --height {height} 
                                    --preprocessor KHot --batch_size 16 --n_iterations 100000 
                                    --lr 0.001 --lr_Z 0.1 --learn_PB --tie_PB {no_cuda}
                                    {use_tb} {use_baseline} {v2} --wandb {wandb} --seed {seed} 
                                    --validation_samples {validation_samples} --validation_interval 100 
                                    --validate_with_training_examples"""
                script_to_run = script_to_run.replace("\n", " ")
                print(f"{counter}, {script_to_run}")
                subprocess.check_output(prefix(job_name) + script_to_run, shell=True)
                counter += 1
