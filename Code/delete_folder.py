import shutil

folder_to_delete = '../../../scratch/tmp/t_liet02/DOTA_exp_AS_AP/train/images'

try:
    shutil.rmtree(folder_to_delete)
    print(f"Ordner '{folder_to_delete}' und sein Inhalt erfolgreich gelöscht.")
except FileNotFoundError:
    print(f"Fehler: Ordner '{folder_to_delete}' nicht gefunden.")
except OSError as e:
    print(f"Fehler beim Löschen von '{folder_to_delete}': {e}")