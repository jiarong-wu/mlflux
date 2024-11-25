#!/bin/bash 

#SBATCH --job-name=ann
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=8GB
#SBATCH --array=0-4  # Create an array with indices 0 to 3

singularity exec --nv --overlay /scratch/jw8736/environments/mlflux.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c  "source /ext3/env.sh; python -u training.py --path=/scratch/jw8736/mlflux/saved_model/final/Mcross5_1/NW_tr2/ --rand=${SLURM_ARRAY_TASK_ID}"


