#!/bin/bash

#SBATCH -o /network/scratch/v/vivianoj/torchgfn/logs/intel/slurm-%j.out
#SBATCH -e /network/scratch/v/vivianoj/torchgfn/logs/intel/slurm-%j.err
#SBATCH -J ddp
#SBATCH --get-user-env
#SBATCH --partition=long
#SBATCH --constraint=milan     # 64 CPU Cores, 1core:12GB optimal ratio.
#SBATCH -N 2                   # Number of nodes to request.
#SBATCH --ntasks-per-node=4    # If you use ntasks-per-node you can scale.
#SBATCH --cpus-per-task=8      # Number of CPUs to run per task.
#SBATCH --time=23:59:59

# Olexa notes:
# TODO: -B 2,16,1
#     --sockets-per-node=2
#     --cores-per-socket=<cores>
#     --threads-per-core=<threads>
#
#     --ntasks-per-socket=4
#

# Initalize the conda environment on the target node, which should automatically set all
# oneapi variables for the user.
source /home/mila/v/vivianoj/miniconda3/bin/activate
conda activate torchgfn_multinode

# System dependent thing - to delete.
#source /swtools/intel/2024.0/oneapi-vars.sh
#source /home/mila/v/vivianoj/intel/oneapi/setvars.sh
#source /home/mila/v/vivianoj/miniconda3/envs/torchgfn/lib/python3.10/site-packages/oneccl_bindings_for_pytorch/env/setvars.sh
#source $(python -c "import oneccl_bindings_for_pytorch as torch_ccl;print(torch_ccl.cwd)")/env/setvars.sh
#echo $(python -c "import oneccl_bindings_for_pytorch as torch_ccl;print(torch_ccl.cwd)")/env/setvars.sh

# Sets the needed environment variables.
#export KMP_AFFINITY=compact,verbose  # TODO: Verify.
export I_MPI_HYDRA_BOOTSTRAP=slurm
export KMP_AFFINITY=none
export MASTER_ADDR=$(hostname  | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}  # Default 1 CPU per task.

echo "***Running run_dist_ht.sh with: ***"
echo " + Master Addr: ${MASTER_ADDR}"
echo " + Slurm Job Num Nodes: ${SLURM_JOB_NUM_NODES}"
echo " + Slurm NodeList: ${SLURM_NODELIST}"

#mpiexec.hydra -np 2 -ppn 2 -l -genv I_MPI_PIN_DOMAIN=[0xFFFF,0xFFFF0000] -genv CCL_WORKER_AFFINITY=32,48 -genv CCL_WORKER_COUNT=1 -genv O     MP_NUM_THREADS=16 python -u train_hypergrid_multinode.py --ndim 8 --height 8 --R0 0.01 --tied --loss TB --n_trajectories 512000 --batch_size 256000
mpiexec.hydra -np 4 -ppn 4 -l -genv CCL_WORKER_COUNT=8 python -u ../train_hypergrid.py --ndim 8 --height 8 --R0 0.01 --tied --loss TB --n_trajectories 512000 --batch_size 256

#./run_dist_ht.sh python -u train_hypergrid_multinode.py --ndim 8 --height 8 --R0 0.01 --tied --loss TB --n_trajectories 512000 --batch_size 256000



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
