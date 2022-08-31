import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--no-cuda", action="store_true", default=False, help="use gpu")
parser.add_argument("--time", default="2:00:0", help="time")
parser.add_argument("--mem", default="12G", help="memory needed for each job")
args = parser.parse_args()

gres = "--gres=gpu:1" if not args.no_cuda else ""


def prefix(job_name):
    return f"sbatch --time {args.time} {gres} --job-name {job_name} -c 4 --mem {args.mem} run.sh "


no_cuda = "--no_cuda"
wandb = "TB_vs_VI_tabular"
validation_samples = 200000
counter = 0
for (ndim, height) in zip([2, 4], [64, 8]):
    for seed in (1, 2):
        for (use_tb, use_baseline) in [
            ("--use_tb", ""),
            ("", "--use_baseline"),
        ]:
            for (learn_PB, tie_PB) in [
                ("--learn_PB", "--tie_PB"),
                ("", ""),
            ]:
                for lr in [0.1, 0.01]:
                    if counter == 0:
                        counter += 1
                        continue
                    job_name = f"{counter}_{wandb}"
                    script_to_run = f"""python compare_TB_to_VI.py --ndim {ndim} --height {height} 
                                        --preprocessor Identity --batch_size 16 --n_iterations 100000 
                                        --lr {lr} --lr_Z 0.1 {learn_PB} {tie_PB} {no_cuda}  --tabular
                                        {use_tb} {use_baseline} --wandb {wandb} --seed {seed} 
                                        --validation_samples {validation_samples} --validation_interval 100 
                                        --validate_with_training_examples"""
                    script_to_run = script_to_run.replace("\n", " ")
                    print(f"{counter}, {script_to_run}")
                    subprocess.check_output(
                        prefix(job_name) + script_to_run, shell=True
                    )
                    counter += 1
