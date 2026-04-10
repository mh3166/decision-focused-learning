#!/bin/bash

# Submit the baseline portfolio run first. It writes model checkpoints that
# submit_warmstart.sh consumes through BASELINE_RUN_ID.

#SBATCH --array=0-199
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=45:00:00
#SBATCH --export=none
#SBATCH --job-name=pg_portfolio_baselines
#SBATCH --mail-user=guptavis@usc.edu


module purge
module load ver/2506
module load gcc/14.3.0
module load python/3.11.14

# Go to experiment directory
cd /scratch1/guptavis/decision-focused-learning/experiments/portfolio_markowitz
python3 train_baselines.py $SLURM_ARRAY_TASK_ID
