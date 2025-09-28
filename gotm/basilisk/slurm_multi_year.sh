#!/bin/bash 

#SBATCH --job-name=GOTM-multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=8GB
#SBATCH --array=0-4  # Create an array with indices 0 to 3

# Define an array of years corresponding to the indices
YEARS=(2011 2012 2013 2015 2016)
# Get the year for the current array job
# Options
# kpp 60 or kpp 10 or kepsilon 60 or kepsilon 10
YEAR=${YEARS[$SLURM_ARRAY_TASK_ID]}
singularity exec /scratch/work/public/singularity/ubuntu-22.04.4.sif /bin/bash -c \
    "bash ./job.sh kpp 60 $YEAR"
