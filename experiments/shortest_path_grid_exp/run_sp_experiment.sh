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
module load ver/2506  
module load gcc/14.3.0
module load python/3.11.14

python3 sp_experiment_slurm.py $SLURM_ARRAY_TASK_ID
