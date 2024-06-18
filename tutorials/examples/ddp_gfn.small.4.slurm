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

./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 512000 --batch_size 256000 &> scaling.out.4.4.512000.256000
#./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 128000 --batch_size 256000 &> scaling.out.4.4.128000.256000
#./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 64000 --batch_size 256000 &> scaling.out.4.4.64000.256000
#./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 32000 --batch_size 256000 &> scaling.out.4.4.32000.256000
#./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories 16000 --batch_size 256000 &> scaling.out.4.4.16000.256000
#./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories  8000 --batch_size 256000 &> scaling.out.4.4.8000.256000
#./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories  4000 --batch_size 256000 &> scaling.out.4.4.4000.256000
#./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories  2000 --batch_size 256000 &> scaling.out.4.4.2000.256000
#./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories  1000 --batch_size 256000 &> scaling.out.4.4.1000.256000
#./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories   512 --batch_size 256000 &> scaling.out.4.4.512.256000
#./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories   256 --batch_size 256000 &> scaling.out.4.4.256.256000
#./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories   128 --batch_size 256000 &> scaling.out.4.4.128.256000
#./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories    64 --batch_size 256000 &> scaling.out.4.4.64.256000
#./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories    32 --batch_size 256000 &> scaling.out.4.4.32.256000
#./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories    16 --batch_size 256000 &> scaling.out.4.4.16.256000
#./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories     8 --batch_size 256000 &> scaling.out.4.4.8.256000
#./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories     4 --batch_size 256000 &> scaling.out.4.4.4.256000
#./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories     2 --batch_size 256000 &> scaling.out.4.4.2.256000
#./run_dist_ht.sh -np 4 -ppn 4 python -u train_hypergrid_multinode.py --ndim 4 --height 64 --R0 0.01 --tied --loss TB --n_trajectories     1 --batch_size 256000 &> scaling.out.4.4.1.256000
