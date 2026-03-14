#!/bin/bash

#SBATCH -J bench_huge
#! Which project should be charged:
#SBATCH -A CASTELNOVO-SL2-CPU
#SBATCH -p icelake-himem
#! Memory calc: 32 bytes/state * 39,747,065,017 states = 1,271,906,080,544 B
#! Min ranks = ceil(1271906080544 / 6,706,022,400) = 190
#! 4 nodes * 76 CPUs = 304 tasks -> 4.18 GB/rank (2.52 GB headroom for MPI buffers)
#SBATCH --nodes=4
#SBATCH --ntasks=304
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6760
#! How much wallclock time will be required?
#SBATCH --time=0:30:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=START,END
#SBATCH --output=track/bench_huge.%A_%a.out
#SBATCH --error=track/bench_huge.%A_%a.error

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-icl              # REQUIRED - loads the basic environment

module load intel/mkl/2020.4
# point to custom HDF5
export LD_LIBRARY_PATH=/home/als217/install/HDF_Group/HDF5/1.14.6/lib:${LD_LIBRARY_PATH}

export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

workdir="/home/als217/rds/hpc-work/QSI/qsi_subspace_ED/hpc_driver"

cd $workdir
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

srun --cpu-bind=cores ../build/bench/bench_apply_mpi \
    ../../lattice_files/pyro_-2,2,2_2,-2,2_2,2,-2.json \
    --basis-type interp
