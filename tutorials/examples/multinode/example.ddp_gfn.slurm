#!/bin/bash

#SBATCH -o /path/to/logs/slurm-%j.out
#SBATCH -e /path/to/logs/slurm-%j.err
#SBATCH -J ddp
#SBATCH --get-user-env
#SBATCH --partition=long
#SBATCH --constraint=milan     # 64 CPU Cores, 1core:12GB optimal ratio.
#SBATCH -N 2                   # Number of nodes to request.
#SBATCH --ntasks-per-node=4    # If you use ntasks-per-node you can scale.
#SBATCH --cpus-per-task=8      # Number of CPUs to run per task.
#SBATCH --time=23:59:59

# Initalize the conda environment on the target node, which should automatically set all
# oneapi variables for the user.
source /path/to/miniconda3/bin/activate
conda activate torchgfn

# Sets the needed environment variables.
export I_MPI_HYDRA_BOOTSTRAP=slurm
export KMP_AFFINITY=none
export MASTER_ADDR=$(hostname  | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}  # Default 1 CPU per task.

echo "***Running run_dist_ht.sh with: ***"
echo " + Master Addr: ${MASTER_ADDR}"
echo " + Slurm Job Num Nodes: ${SLURM_JOB_NUM_NODES}"
echo " + Slurm NodeList: ${SLURM_NODELIST}"

mpiexec.hydra -np 4 -ppn 4 -l -genv CCL_WORKER_COUNT=8 python -u ../train_hypergrid.py --ndim 8 --height 8 --R0 0.01 --tied --loss TB --n_trajectories 512000 --batch_size 256