import os
from ultralytics import YOLO

# Umgebung lesen
perm_set = os.environ["PERM_SET"]
fold_id = os.environ["FOLD_ID"]
data_yaml = os.environ["DATA_YAML"]

# Pfade zusammenbauen
model_path = f"/scratch/tmp/t_liet02/cross_validation/{perm_set}/fold{fold_id}/weights/best.pt"
output_csv = f"/scratch/tmp/t_liet02/new_val_detect/{perm_set}/fold{fold_id}/confusion_matrix.csv"

# Modell laden
model = YOLO(model_path)

# Validierung
results = model.val(data=data_yaml, plots=True)

# Confusion Matrix als CSV speichern
conf_matrix_df = results.confusion_matrix.to_df()
conf_matrix_df.to_csv(output_csv, index=False)

print("Confusion Matrix gespeichert in:", output_csv)
