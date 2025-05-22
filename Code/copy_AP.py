import os
import shutil

source_dir = '../../../scratch/tmp/t_liet02/airbus_datasets/train_v2'
destination_dir = '../../../scratch/tmp/t_liet02/DOTA_exp_AS_AP/train/images'

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