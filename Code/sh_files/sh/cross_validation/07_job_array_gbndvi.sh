#!/bin/bash

#SBATCH --job-name=gbndvi_arr
#SBATCH --export=NONE
#SBATCH --nodes=1
#SBATCH --gres=gpu:4 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --partition=gpua100,gpu4090,gpu2080 
#SBATCH --time=00:10:00
#SBATCH --array=0-4
#SBATCH --output=output/new_val/gbndvi/fold%a.dat
#SBATCH --mail-type=None
#SBATCH --mail-user=t_liet02@uni-muenster.de
#SBATCH --nice=100

# Modulumgebung und virtuelle Umgebung aktivieren
module purge
module load uv
source /scratch/tmp/t_liet02/envs/__mt_uv__/.venv/bin/activate

# Array-abhaengiger Ordnername und YAML-Datei
PERM_SET="gbndvi"
FOLD_ID=${SLURM_ARRAY_TASK_ID}
DATA_YAML="/scratch/tmp/t_liet02/data/cross_validation/${PERM_SET}/fold${FOLD_ID}/data.yaml"

echo "Starte YOLOv9-OBB Training fuer Fold ${FOLD_ID}"
echo "Verwende YAML: ${DATA_YAML}"

# Training mit yolo CLI starten
#yolo detect val \
#    model=/scratch/tmp/t_liet02/cross_validation/${PERM_SET}/fold${FOLD_ID}/weights/best.pt \
#    data=${DATA_YAML} \
#    split=val \
#    imgsz=1024 \
#    batch=4 \
#    device=[0] \
#    project=/scratch/tmp/t_liet02/new_val_detect/${PERM_SET}/fold${FOLD_ID} \
#    save_txt=True \
#    save_json=True \
#    save_conf=True \
#    exist_ok=True


export PERM_SET
export FOLD_ID
export DATA_YAML

python master_thesis/yolo_val_csv.py || { echo "Python-Skript fehlgeschlagen"; exit 1; }

    