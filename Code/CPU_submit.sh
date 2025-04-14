#!/bin/bash
#SBATCH --job-name=heatmap_test
#SBATCH --export=NONE               # Start with a clean environment
#SBATCH --cpus-per-task=8      
#SBATCH --mem=100G                   # how much memory is needed per node (units can be: K, M, G, T)
#SBATCH --partition=normal          # on which partition to submit the job
#SBATCH --time=24:00:00             # the max wallclock time (time limit your job will run)
#SBATCH --output=SHB_Job/outputs/create_heatmaps.dat         # the file where output is written to (stdout & stderr)
#SBATCH --mail-type=END,FAIL             # receive an email when your job starts, finishes normally or is aborted
#SBATCH --mail-user=t_liet02@uni-muenster.de # your mail address
#SBATCH --nice=100
 
module purge
module load palma/2021a Miniconda3/4.9.2

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda deactivate
conda activate envs/_condaENV2

python SHB_Job/scripts/multiple_object_tracking.py

#python scripts/preprocessing/build_dataset.py

