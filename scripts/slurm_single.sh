#!/bin/bash 

#SBATCH --job-name=ann
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=8GB

singularity exec --nv --overlay /scratch/jw8736/environments/mlflux.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c  "source /ext3/env.sh; python -u training.py --path=/home/jw8736/mlflux/saved_model/final/SH8_2/tr2/ --rand=3"


