#!/bin/bash

#SBATCH --job-name=gpu_dota_test
#SBATCH --export=NONE             # Start with a clean environment
#SBATCH --nodes=2                 # the number of nodes you want to reserve
#SBATCH --gres=gpu:8              # 8 GPUs per node (total 16 GPUs)
#SBATCH --ntasks-per-node=1       # Important: One task per node to manage the distributed launch
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --partition=gpu2080
#SBATCH --time=96:00:00
#SBATCH --output=output/train_DOTA_EXP_test.dat
#SBATCH --mail-type=start,end
#SBATCH --mail-user=t_liet02@uni-muenster.de
#SBATCH --nice=100

module purge
module load uv

source /scratch/tmp/t_liet02/envs/__mt_uv__/.venv/bin/activate

# --- WICHTIG: Distributed Training Setup ---
# Anzahl der GPUs pro Knoten
GPUS_PER_NODE=8
# Gesamtzahl der GPUs
TOTAL_GPUS=$((SLURM_NNODES * GPUS_PER_NODE))

# WICHTIG: Ultralytics mit DDP (DistributedDataParallel) starten
# Hier nutzen wir `yolo` direkt und lassen es die DDP-Logik handhaben.
# Das `device` Argument sollte jetzt leer sein, da Ultralytics selbst die GPUs verwaltet.
# Alternativ könnte man auch '0,1,2,3,4,5,6,7' angeben, aber oft ist leer besser.
# Das Schlüsselwort `--world_size` wird von Ultralytics intern basierend auf der Umgebung gesetzt.

# Sie müssen hier den Befehl 'srun' verwenden, um den Befehl auf allen zugewiesenen Knoten auszuführen.
# 'srun' stellt sicher, dass die Umgebungsvariablen für verteiltes Training korrekt gesetzt werden.
srun yolo train \
    data=/scratch/tmp/t_liet02/DOTA_exp_AP_AS/data.yaml \
    model=/scratch/tmp/t_liet02/yolov9u/ultralytics/cfg/models/v9/yolov9-obb.yaml \
    project=/scratch/tmp/t_liet02/yolov9u_runs \
    epochs=200 \
    patience=15 \
    device="" \ # Wichtig: Gerät nicht manuell angeben oder '0,1,2,3,4,5,6,7' wenn Sie DDP auf einem Knoten nutzen
    batch=16 \ # Batch-Size ist pro GPU (Gesamt-Batch wird 16 * TOTAL_GPUS)
    imgsz=1024 \
    name='DOTA_Exp_EP90_PAT15_test_distributed' \
    exist_ok=True \
    pretrained=False