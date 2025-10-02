import os
import shutil

source_dir = '../../../scratch/tmp/t_liet02/airbus_datasets/train_v2'
destination_dir = '../../../scratch/tmp/t_liet02/DOTA_exp_AS_AP/train/images'

"""
Copies all image files from a source directory to a destination directory, ensuring the destination exists.

- Creates the destination directory if it does not exist.
- Iterates over all files in the source directory.
- Copies each file to the destination directory using shutil.copy2 (preserving metadata).
- Prints progress for each copied file.
- Handles and prints errors if files cannot be copied or if the source directory does not exist.

Variables:
    source_dir (str): Path to the source directory containing images.
    destination_dir (str): Path to the destination directory for copied images.
"""


try:
    # Stelle sicher, dass der Zielordner existiert
    os.makedirs(destination_dir, exist_ok=True)
    print(f"Zielordner '{destination_dir}' wurde erstellt oder existiert bereits.")

    # Hole die Liste aller Dateien im Quellordner
    all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    total_files = len(all_files)
    copied_count = 0

    # Gehe durch alle Dateien im Quellordner
    for filename in all_files:
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(destination_dir, filename)

        try:
            shutil.copy2(source_path, destination_path)
            copied_count += 1
            print(f"Kopiere Bild {copied_count} von {total_files}: '{filename}'")
        except Exception as e:
            print(f"Fehler beim Kopieren von '{filename}': {e}")

    print("Kopiervorgang abgeschlossen.")

except FileNotFoundError:
    print(f"Fehler: Quellordner '{source_dir}' nicht gefunden.")
except Exception as e:
    print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")