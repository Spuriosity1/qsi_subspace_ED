#!/bin/bash

#SBATCH -J ed_sub_detail
#! Which project should be charged:
#SBATCH -A CASTELNOVO-SL2-CPU
#SBATCH -p icelake
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total? (<= nodes*76)
#! The Ice Lake (icelake) nodes have 76 CPUs (cores) each and
#! 3380 MiB of memory per CPU.
#SBATCH --ntasks=1
#! How much wallclock time will be required?
#SBATCH --time=12:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=END,TIME_LIMIT_80
#SBATCH --output=track/phasedia_detail.%A_%a.out
#SBATCH --error=track/phasedia_detail.%A_%a.error


. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-icl              # REQUIRED - loads the basic environment
module load python/3.11.0-icl



export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK


# load the custom py env
. /home/als217/rds/hpc-work/QSI/my_python/bin/activate


workdir="/home/als217/rds/hpc-work/QSI/qsi_subspace_ED/hpc_driver"

cd $workdir
echo -e "Changed directory to `pwd`.\n"


JOBID=$SLURM_JOB_ID
TASKID=$SLURM_ARRAY_TASK_ID

echo -e "TaskID: $TASKID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

#if [ "$SLURM_JOB_NODELIST" ]; then
#        #! Create a machine file:
#        export NODEFILE=`generate_pbs_nodefile`
#        cat $NODEFILE | uniq > machine.file.$JOBID
#        echo -e "\nNodes allocated:\n================"
#        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
#fi
#

