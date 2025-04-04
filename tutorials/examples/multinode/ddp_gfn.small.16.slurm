#!/bin/bash

#SBATCH -o job.%j.out
#SBATCH -J ddp
#SBATCH --get-user-env
#SBATCH --partition=spr
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=112
#SBATCH --time=00:30:00

source /swtools/intel/2024.0/oneapi-vars.sh
export I_MPI_HYDRA_BOOTSTRAP=slurm

source ~/installs/anaconda3/bin/activate  gfn_may14
export KMP_AFFINITY=compact,verbose
export OMP_NUM_THREADS=56
export MASTER_ADDR=$(hostname  | head -n 1)
echo $MASTER_ADDR
echo $SLURM_JOB_NUM_NODES
echo $SLURM_NODELIST

./run_dist_ht.sh -np 16 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 512000 --batch_size 128 &> scaling.out.16.4.512000.128
#./run_dist_ht.sh -np 16 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 1024000 --batch_size 256000 &> scaling.out.16.4.1024000.256000
#./run_dist_ht.sh -np 16 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 2048000 --batch_size 256000 &> scaling.out.16.4.2048000.256000
#./run_dist_ht.sh -np 16 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 4096000 --batch_size 256000 &> scaling.out.16.4.4096000.256000
#./run_dist_ht.sh -np 16 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 8192000 --batch_size 256000 &> scaling.out.16.4.8192000.256000
