#!/bin/bash
#SBATCH --job-name=DMRGjl
#SBATCH --output=slurmlogs/tmp/slurm-%A.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=24:00:00

module purge

module load anaconda3/2024.2
#module load cudatoolkit/12.4
module load julia/1.10.2
conda activate julia

julia main.jl


today=$(date +%Y-%m-%d)
filen=slurm-${SLURM_JOB_ID}.out

mkdir -p slurmlogs/$today
mv slurmlogs/tmp/$filen slurmlogs/$today/$filen
