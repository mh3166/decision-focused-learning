#!/bin/bash

# Submit this only after submit_baselines.sh has completed. BASELINE_RUN_ID
# must identify that baseline SLURM_ARRAY_JOB_ID so train_warmstart.py can load
# the saved baseline checkpoints.

#SBATCH --array=0-599%100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=45:00:00
#SBATCH --export=none
#SBATCH --job-name=pg_portfolio_warmstart
#SBATCH --mail-user=guptavis@usc.edu

set -euo pipefail

module purge
module load ver/2506

module load gurobi/11.0.2
module load gcc/14.3.0

module load python/3.11.14
python3 -c "import gurobipy; print('gurobipy', gurobipy.gurobi.version())"

# Go to experiment directory
cd /scratch1/guptavis/decision-focused-learning/experiments/portfolio_markowitz

# Required by train_warmstart.py. Update this to the SLURM_ARRAY_JOB_ID
# from the train_baselines.py baseline run before submitting.
export BASELINE_RUN_ID=REPLACE_WITH_BASELINE_RUN_ID

python3 train_warmstart.py "$SLURM_ARRAY_TASK_ID"
