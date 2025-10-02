import shutil

"""
Deletes a folder and all its contents using shutil.rmtree.

- Attempts to remove the specified folder and its contents.
- Prints a success message if deletion is successful.
- Handles FileNotFoundError if the folder does not exist.
- Handles and prints other OS errors if deletion fails.

Variables:
    folder_to_delete (str): Path to the folder to be deleted.
"""

folder_to_delete = '../../../scratch/tmp/t_liet02/DOTA_exp_AS_AP/train/images'

try:
    shutil.rmtree(folder_to_delete)
    print(f"Ordner '{folder_to_delete}' und sein Inhalt erfolgreich gelöscht.")
except FileNotFoundError:
    print(f"Fehler: Ordner '{folder_to_delete}' nicht gefunden.")
except OSError as e:
    print(f"Fehler beim Löschen von '{folder_to_delete}': {e}")