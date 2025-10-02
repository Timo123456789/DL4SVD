import os
from ultralytics import YOLO

"""
This script performs validation of a YOLO model on a specified dataset and saves the resulting confusion matrix as a CSV file.

Process:
    - Reads environment variables to determine the permutation set, fold ID, and data YAML file.
    - Constructs the model path and output CSV path based on these variables.
    - Loads the YOLO model from the specified weights file.
    - Runs validation using the provided data YAML file, generating plots.
    - Extracts the confusion matrix from the validation results and saves it as a CSV file.
    - Prints the location of the saved confusion matrix.

Environment Variables:
    PERM_SET (str): Name of the permutation set (used for path construction).
    FOLD_ID (str or int): Fold number for cross-validation (used for path construction).
    DATA_YAML (str): Path to the data YAML file describing the dataset.

Outputs:
    - Confusion matrix CSV file saved to the specified output path.
    - Console message indicating where the confusion matrix was saved.
"""

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
