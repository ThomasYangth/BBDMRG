#!/bin/bash

# Get today's date in the format YYYY-MM-DD
today=$(date +%Y-%m-%d)

# Create the directory for logs
mkdir -p slurmlogs/$today

# Submit the actual job script
sbatch single_runjl.slurm
