#!/bin/bash

#SBATCH --job-name=orFals_Ep1500
#SBATCH --export=NONE               # Start with a clean environment
#SBATCH --nodes=1                   # the number of nodes you want to reserve
#SBATCH --gres=gpu:4 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8       
#SBATCH --mem=45G                   # how much memory is needed per node (units can be: K, M, G, T)
#SBATCH --partition=gpu4090,gpuh200,gpu3090,gpua100,gpuhgx,gpu2080      # on which partition to submit the job
#SBATCH --time=24:00:00             # the max wallclock time (time limit your job will run)
#SBATCH --output=orFals_Ep1500.dat         # the file where output is written to (stdout & stderr)
#SBATCH --mail-type=Start,End          # receive an email when your job starts, finishes normally or is aborted
#SBATCH --mail-user=t_liet02@uni-muenster.de # your mail address
#SBATCH --nice=100
 
module purge
module load palma/2021a Miniconda3/4.9.2

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda deactivate
conda activate envs/__mt__

python master_thesis/yolov9/train_dual.py --data /scratch/tmp/t_liet02/data/fold1_oriented_false/data.yaml --cfg /home/t/t_liet02/master_thesis/yolov9/models/detect/yolov9.yaml --workers 8 --epochs 1500 --patience 150 --device 0,1,2,3 --batch-size 4 --img 1024 --name orFalse_Ep1500


# python master_thesis/yolov9/train_dual.py \
#     --data /scratch/tmp/t_liet02/data/fold1_oriented_false/data.yaml \
#     --cfg /home/t/t_liet02/master_thesis/yolov9/models/detect/yolov9.yaml \
#     --workers 8 \
#     --epochs 500 \
#     --patience 50 \
#     --device 0,1,2,3 \
#     --batch-size 4 \
#     --img 1024 \  
    
  
    #--hyp data/hyps/hyp.scratch-high.yaml   
#yolo train data=../../../scratch/tmp/t_liet02/data/fold1_oriented_False/data.yaml model=master_thesis/yolov9u/ultralytics/cfg/models/v9/yolov9e.yaml name='yolov9_oriented_false' epochs=2 device=cpu imgsz=1024

#python master_thesis/preproc_folds.py
