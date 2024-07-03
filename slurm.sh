#!/bin/sh
#提交单个作业
#SBATCH -J-name=InstantAvatar
#SBATCH --ntasks=16

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH -p A6000
#SBATCH -A gpu2002

#SBATCH --time=240:00:00
#SBATCH --output=out.job
#SBATCH --error=error.job
#SBATCH --mail-type=ALL
#SBATCH --mail-user=address

# module load /share/home/gpu2002/miniconda3
conda activate instantAvatar

# HYDRA_FULL_ERROR=1 sh bash/run-demo.sh
HYDRA_FULL_ERROR=1 sh bash/run-demo.sh
