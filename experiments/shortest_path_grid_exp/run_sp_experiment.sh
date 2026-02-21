#!/bin/bash

#SBATCH --array=0-799
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=45:00:00
#SBATCH --export=none 
#SBATCH --job-name=pg_sp_experiment_baselines
#SBATCH --mail-user=guptavis@usc.edu


module purge
module load legacy/CentOS7
module load gcc/11.3.0
module load python/3.9.12

python3 experiments/shortest_path_grid_exp/sp_experiment_slurm.py $SLURM_ARRAY_TASK_ID
