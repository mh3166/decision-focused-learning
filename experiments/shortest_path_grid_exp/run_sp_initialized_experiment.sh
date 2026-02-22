#!/bin/bash

  #SBATCH --array=0-2399%100
  #SBATCH --ntasks=1
  #SBATCH --cpus-per-task=1
  #SBATCH --mem-per-cpu=4GB
  #SBATCH --time=45:00:00
  #SBATCH --export=none
  #SBATCH --job-name=pg_sp_experiment_initialized
  #SBATCH --mail-user=guptavis@usc.edu

  module purge
  module load ver/2506
  module load gcc/14.3.0
  module load python/3.11.14

  # Go to repo root
  cd /scratch1/guptavis/decision-focused-learning/experiments/shortest_path_grid_exp
  python3 sp_experiment_initialized_slurm.py $SLURM_ARRAY_TASK_ID