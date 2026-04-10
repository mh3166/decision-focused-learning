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

module purge
module load ver/2506
module load gcc/14.3.0
module load python/3.11.14

# Go to experiment directory
cd /scratch1/guptavis/decision-focused-learning/experiments/portfolio_markowitz

# Required by train_warmstart.py. Set this to the SLURM_ARRAY_JOB_ID
# from the train_baselines.py baseline run.
: "${BASELINE_RUN_ID:?Please set BASELINE_RUN_ID to the portfolio baseline run id.}"

python3 train_warmstart.py $SLURM_ARRAY_TASK_ID
