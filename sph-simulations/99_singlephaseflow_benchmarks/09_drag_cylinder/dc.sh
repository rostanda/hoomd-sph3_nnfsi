#!/bin/bash
#SBATCH --job-name=dc              # Job name
#SBATCH --mail-type=BEGIN,END,FAIL        # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=david.krach@mib.uni-stuttgart.de
#SBATCH --ntasks=4                     # Number of MPI tasks (i.e. processes)
#SBATCH --nodes=1                         # Maximum number of nodes to be allocated
#SBATCH --distribution=cyclic:cyclic      # Distribute tasks cyclically first among nodes and then among sockets within a node
#SBATCH --mem=25G
#SBATCH --time=24:00:00                   # Wall time limit (days-hrs:min:sec)
#SBATCH --output=dc_%j.log         # Path to the standard output and error files relative to the working directory
#SBATCH --error=dc_%j.err          # Path to the standard error file
#SBATCH --partition=cpu                   # put the job into the cpu partition

# Ensure that all of the cores are on the same Inifniband network
#SBATCH --contiguous

module load openmpi/5.0.1_gcc-12.2_cuda-12.3
module load gcc/12.2.0

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo "JobID = $SLURM_JOB_ID"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pushd "${SCRIPT_DIR}" > /dev/null
python3 create_domain.py
INIT=$(ls -t *_init.gsd | head -1)
/usr/local.nfs/software/openmpi/5.0.1_gcc-12.2_cuda-12.3/bin/mpirun -np 4 python3 run_adami_cylinder2.py 96 "${INIT}"
popd > /dev/null
