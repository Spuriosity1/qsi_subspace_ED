#!/bin/bash
#SBATCH -J gen_basis
#SBATCH -A CASTELNOVO-SL2-CPU
#SBATCH -p cclake
#SBATCH --nodes=10
#! number of MPI tasks
#SBATCH --ntasks=300
#SBATCH --mail-user=alaric@zedat.fu-berlin.de   # Email to which notifications will be sent
#! How much wallclock time will be required?
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3000
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=track/gen_basis.%A_%a.out
#SBATCH --error=track/gen_basis.%A_%a.error
#SBATCH --chdir=/home/als217/rds/hpc-work/QSI/qsi_subspace_ED

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-icl              # REQUIRED - loads the basic environment


export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


echo -e "Working in to `pwd`.\n"

JOBID=$SLURM_JOB_ID
TASKID=$SLURM_ARRAY_TASK_ID

TMP_PATH="../tmp/sbsearch108"
EXE_DIR="build/components/build_basis"

echo -e "TaskID: $TASKID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

echo -e "Working in to `pwd`.\n"

JOBID=$SLURM_JOB_ID
TASKID=$SLURM_ARRAY_TASK_ID
latfile="pyro_3,0,0_0,3,0_0,0,3.json"

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi


mkdir -p $TMP_PATH
rm $TMP_PATH/*
CMD="mpirun $EXE_DIR/sbsearch_mpi ../lattice_files/$latfile -o $TMP_PATH"

echo "$CMD"
$CMD

