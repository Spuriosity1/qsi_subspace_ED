#!/bin/bash

#SBATCH -J bampi3
#! Which project should be charged:
#SBATCH -A CASTELNOVO-SL3-CPU
#SBATCH -p icelake-himem
#! How many whole nodes should be allocated?
#SBATCH --nodes=3
#! How many (MPI) tasks will there be in total? (<= nodes*76)
#! The Ice Lake (icelake) nodes have 76 CPUs (cores) each and
#! 3380 MiB of memory per CPU.
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6760
#! How much wallclock time will be required?
#SBATCH --time=0:20:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=START,END
#SBATCH --output=track/bampi3.%A_%a.out
#SBATCH --error=track/bampi3.%A_%a.error

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-icl              # REQUIRED - loads the basic environment

module load intel/mkl/2020.4
# point to custom HDF5
export LD_LIBRARY_PATH=/home/als217/install/HDF_Group/HDF5/1.14.6/lib:${LD_LIBRARY_PATH}

export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

workdir="/home/als217/rds/hpc-work/QSI/test_qsi_subspace_ED/hpc_driver"

cd $workdir
echo -e "Changed directory to `pwd`.\n"


JOBID=$SLURM_JOB_ID

echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

# srun --cpu-bind=cores ../build/bench/bench_apply_mpi ../../lattice_files/pyro_2,0,0_0,3,0_0,0,3.json 
srun --cpu-bind=cores ../build/bench/bench_apply_mpi ../../lattice_files/pyro_2,0,0_0,3,0_0,0,4.json --sector basis_s12.12.12.12 

