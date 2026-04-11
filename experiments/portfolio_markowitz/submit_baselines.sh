#!/bin/bash

# Submit the baseline portfolio run first. It writes model checkpoints that
# submit_warmstart.sh consumes through BASELINE_RUN_ID.

# Original full baseline array:
#SBATCH --array=0-199
#
# TEMPORARY SINGLE-CONFIG RERUN:
# Restrict this batch submission to sim=13, which corresponds to n=200,
# trial=13 in the original baseline enumeration.
#SBATCH --array=13-13
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=45:00:00
#SBATCH --export=none
#SBATCH --job-name=pg_portfolio_baselines
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
python3 train_baselines.py "$SLURM_ARRAY_TASK_ID"
