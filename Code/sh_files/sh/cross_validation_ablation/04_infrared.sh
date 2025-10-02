#!/bin/bash

#SBATCH --job-name=Abl_IR_arr
#SBATCH --export=NONE
#SBATCH --nodes=1
#SBATCH --gres=gpu:4 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --partition=gpua100,gpu4090,gpua100,gpuhgx,gpuh200
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=output/abl/ir/fold%a.dat
#SBATCH --mail-type=None
#SBATCH --mail-user=t_liet02@uni-muenster.de
#SBATCH --nice=100

# Modulumgebung und virtuelle Umgebung aktivieren
module purge
module load uv
source /scratch/tmp/t_liet02/envs/__mt_uv__/.venv/bin/activate

# Array-abhängiger Ordnername und YAML-Datei
PERM_SET="ir"
FOLD_ID=${SLURM_ARRAY_TASK_ID}
DATA_YAML="/scratch/tmp/t_liet02/data/cross_validation_ablation/${PERM_SET}/fold${FOLD_ID}/data.yaml"

echo "Starte YOLOv9-OBB Training für Fold ${FOLD_ID}"
echo "Verwende YAML: ${DATA_YAML}"


yolo train \
   data=${DATA_YAML} \
   model=/scratch/tmp/t_liet02/yolov9u/ultralytics/cfg/models/v9/yolov9-obb.yaml \
   epochs=500 \
   device=[0,1,2,3] \
   batch=4 \
   workers=12 \
   imgsz=1024 \
   project= /scratch/tmp/t_liet02/cross_validation_ablation/${PERM_SET} \
   name="fold${FOLD_ID}" \
   exist_ok=True \
   patience=0 \
   pretrained=False


export PERM_SET
export FOLD_ID
export DATA_YAML

python master_thesis/yolo_val_csv.py || { echo "Python-Skript fehlgeschlagen"; exit 1; }

    