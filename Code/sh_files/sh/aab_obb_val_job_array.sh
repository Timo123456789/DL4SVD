#!/bin/bash
#SBATCH --job-name=aab_obb_arr
#SBATCH --export=NONE
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --partition=gpu2080,gputitanrtx,gpu3090,gpu4090
#SBATCH --time=01:00:00
#SBATCH --array=0-4
#SBATCH --output=output/oob_aab_val/fold%a.dat
#SBATCH --mail-type=None
#SBATCH --mail-user=t_liet02@uni-muenster.de
#SBATCH --nice=100

# Modulumgebung und virtuelle Umgebung aktivieren
module purge
module load uv
source /scratch/tmp/t_liet02/envs/__mt_uv__/.venv/bin/activate

# Array-abhängiger Ordnername und YAML-Datei
PERM_SET="aab"
FOLD_ID=${SLURM_ARRAY_TASK_ID}
DATA_YAML="/scratch/tmp/t_liet02/data/cross_validation/${PERM_SET}/fold${FOLD_ID}/data.yaml"

echo "Starte YOLOv9-OBB Training für Fold ${FOLD_ID}"
echo "Verwende YAML: ${DATA_YAML}"


export PERM_SET
export FOLD_ID
export DATA_YAML

python master_thesis/yolo_val_aab_oob_csv.py || { echo "Python-Skript fehlgeschlagen"; exit 1; }

PERM_SET="aab_old"
FOLD_ID=${SLURM_ARRAY_TASK_ID}
DATA_YAML="/scratch/tmp/t_liet02/data/cross_validation/${PERM_SET}/fold${FOLD_ID}/data.yaml"

echo "Starte YOLOv9-OBB Training für Fold ${FOLD_ID}"
echo "Verwende YAML: ${DATA_YAML}"


export PERM_SET
export FOLD_ID
export DATA_YAML

python master_thesis/yolo_val_aab_oob_csv.py || { echo "Python-Skript fehlgeschlagen"; exit 1; }


PERM_SET="obb"
FOLD_ID=${SLURM_ARRAY_TASK_ID}
DATA_YAML="/scratch/tmp/t_liet02/data/cross_validation/${PERM_SET}/fold${FOLD_ID}/data.yaml"

echo "Starte YOLOv9-OBB Training für Fold ${FOLD_ID}"
echo "Verwende YAML: ${DATA_YAML}"


export PERM_SET
export FOLD_ID
export DATA_YAML

python master_thesis/yolo_val_aab_oob_csv.py || { echo "Python-Skript fehlgeschlagen"; exit 1; }
    