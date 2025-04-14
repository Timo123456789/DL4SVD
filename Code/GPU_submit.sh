#!/bin/bash

#SBATCH --job-name=Y9_CarDet_100EP
#SBATCH --export=NONE               # Start with a clean environment
#SBATCH --nodes=1                   # the number of nodes you want to reserve
#SBATCH --gres=gpu:2 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8       
#SBATCH --mem=45G                   # how much memory is needed per node (units can be: K, M, G, T)
#SBATCH --partition=gputitanrtx,gpua100,gpuhgx,gpu3090          # on which partition to submit the job
#SBATCH --time=100:00:00             # the max wallclock time (time limit your job will run)
#SBATCH --output=Y9_CD_100EP_temp.dat         # the file where output is written to (stdout & stderr)
#SBATCH --mail-type=END,FAIL            # receive an email when your job starts, finishes normally or is aborted
#SBATCH --mail-user=t_liet02@uni-muenster.de # your mail address
#SBATCH --nice=100
 
module purge
module load foss/2019a
module load Python/3.7.2

python train_dual.py \
    --workers 8 \
    --epochs 1 \
    --device [0,1] \
    --batch-size 2 \
    --data car/car.yaml \
    --img 640 \
    --cfg models/detect/yolov9-t.yaml \
    --name yolov9-car-detector_100EP \
    --hyp data/hyps/hyp.scratch-high.yaml   