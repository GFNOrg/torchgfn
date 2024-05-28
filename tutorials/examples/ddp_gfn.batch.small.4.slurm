#!/bin/bash

#SBATCH -o job.%j.out
#SBATCH -J ddp
#SBATCH --get-user-env
#SBATCH --partition=spr
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=112
#SBATCH --time=00:60:00

source /swtools/intel/2024.0/oneapi-vars.sh
export I_MPI_HYDRA_BOOTSTRAP=slurm

source ~/installs/anaconda3/bin/activate  gfn_may14
export KMP_AFFINITY=compact,verbose
export OMP_NUM_THREADS=56
export MASTER_ADDR=$(hostname  | head -n 1)
echo $MASTER_ADDR
echo $SLURM_JOB_NUM_NODES
echo $SLURM_NODELIST

./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 256000 --batch_size  32 &> batch.out.4.4.256000.32
./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 256000 --batch_size  64 &> batch.out.4.4.256000.64
./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 256000 --batch_size 128 &> batch.out.4.4.256000.128
./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 256000 --batch_size 256 &> batch.out.4.4.256000.256
./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 256000 --batch_size 512 &> batch.out.4.4.256000.512
./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 256000 --batch_size  1000 &> batch.out.4.4.256000.1000
./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 256000 --batch_size  2000 &> batch.out.4.4.256000.2000
./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 256000 --batch_size 4000 &> batch.out.4.4.256000.4000
./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 256000 --batch_size 8000 &> batch.out.4.4.256000.8000
./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 256000 --batch_size 16000 &> batch.out.4.4.256000.16000
./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 256000 --batch_size 32000 &> batch.out.4.4.256000.32000
./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 256000 --batch_size 64000 &> batch.out.4.4.256000.64000
./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 256000 --batch_size 128000 &> batch.out.4.4.256000.128000
./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 256000 --batch_size 256000 &> batch.out.4.4.256000.256000
./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 512000 --batch_size 32000 &> batch.out.4.4.512000.32000
./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 512000 --batch_size 64000 &> batch.out.4.4.512000.64000
./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 512000 --batch_size 128000 &> batch.out.4.4.512000.128000
./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 512000 --batch_size 256000 &> batch.out.4.4.512000.256000
./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 512000 --batch_size 512000 &> batch.out.4.4.512000.512000
