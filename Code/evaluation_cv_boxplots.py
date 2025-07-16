import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
    # Manuelle Legenden mit allen Folds
from matplotlib.lines import Line2D
def main():
    all_sets = ["rgbir", "rgb", "irgb", "rirb", "rgir", "gbndvi", "rgbndvi"]
    set_arr = {s: s for s in all_sets}
    window_size = 20

    # Nur mAP@50-95 wird aktiviert
    bool_arr = {
        "train/box_loss": False,
        "train/cls_loss": False,
        "train/dfl_loss": False,
        "precision": False,
        "recall": False,
        "mAP@50": False,
        "mAP@50-95": True,
        "val_box_loss": False,
        "val/cls_ls": False,
        "val/dfl_loss": False,
        "lr/pg0": False,
        "lr/pg1": False,
        "lr/pg2": False,
        "f1_score": False,
        "max_val": 1500,
    }

    set_paths_dict = load_all_sets(set_arr)
    create_boxplot_from_sets(set_paths_dict, window_size, bool_arr)

    

def create_boxplot_from_sets(set_paths_dict, window_size, bool_arr):
    all_combined_dfs = []

    for set_name, fold_paths in set_paths_dict.items():
        if not fold_paths:
            print(f"Set {set_name} enthält keine gültigen Dateien – wird übersprungen.")
            continue

        all_folds, string_title = get_data_Folds(fold_paths, bool_arr, set_name)

        # ➕ Drucke höchsten mAP pro Fold
        for df in all_folds:
            fold_name = df['Fold'].iloc[0] if not df.empty else "Unbekannt"
            if not df.empty:
                max_map = df['mAP'].max()
                print(f"Max {string_title} für {set_name} – {fold_name}: {max_map:.4f}")


        combined_df = pd.concat(all_folds, ignore_index=True)
        combined_df['Type'] = set_name

        if bool_arr.get('max_val') is not None:
           combined_df = combined_df.sort_values(by='mAP', ascending=False).head(bool_arr['max_val'])

        all_combined_dfs.append(combined_df)

    if not all_combined_dfs:
        print("Keine gültigen Daten gefunden.")
        return

    final_df = pd.concat(all_combined_dfs, ignore_index=True)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=final_df, x='Type', y='mAP', palette='Set2')
    plt.title(f'{string_title} – Boxplot over all Epochs and Folds')
    plt.xlabel('Model')
    plt.ylabel(string_title)
    plt.grid(True, linestyle=':', alpha=0.5)
    sns.despine()
    plt.savefig("Model_Comparison.svg", format="svg", transparent=True)
    plt.tight_layout()
    plt.show()


  





def get_data_Folds(fold_paths, bool_arr, set_string):

    # Für alle Elemente in bool_arr eine if-Abfrage ergänzen
    if bool_arr.get("train/box_loss", False):
        map_spalte = 'train/box_loss'
        string_title = "train/box_loss"
    elif bool_arr.get("train/cls_loss", False):
        map_spalte = 'train/cls_loss'
        string_title = "train/cls_loss"
    elif bool_arr.get("train/dfl_loss", False):
        map_spalte = 'train/dfl_loss'
        string_title = "train/dfl_loss"
    elif bool_arr.get("precision", False):
        map_spalte = 'metrics/precision(B)'
        string_title = "precision(B)"
    elif bool_arr.get("recall", False):
        map_spalte = 'metrics/recall(B)'
        string_title = "recall(B)"
    elif bool_arr.get("mAP@50", False):
        map_spalte = 'metrics/mAP50(B)'
        string_title = "mAP50(B)"
    elif bool_arr.get("mAP@50-95", False):
        map_spalte = 'metrics/mAP50-95(B)'
        string_title = "mAP50-95(B)"
    elif bool_arr.get("val_box_loss", False):
        map_spalte = 'val/box_loss'
        string_title = "val/box_loss"
    elif bool_arr.get("val/cls_ls", False):
        map_spalte = 'val/cls_loss'
        string_title = "val/cls_loss"
    elif bool_arr.get("val/dfl_loss", False):
        map_spalte = 'val/dfl_loss'
        string_title = "val/dfl_loss"
    elif bool_arr.get("lr/pg0", False):
        map_spalte = 'lr/pg0'
        string_title = "lr/pg0"
    elif bool_arr.get("lr/pg1", False):
        map_spalte = 'lr/pg1'
        string_title = "lr/pg1"
    elif bool_arr.get("lr/pg2", False):
        map_spalte = 'lr/pg2'
        string_title = "lr/pg2"
    elif bool_arr.get("f1_score", False):
        map_spalte = None  # da wir F1 manuell berechnen
        string_title = "F1-Score"
    else:
        print("Error bei Spaltenbezeichnung")
   
    epoch_spalte = 'epoch'
    alle_folds_df = []
     # === Alle mAP-Werte + Epochen pro Fold extrahieren ===
    for i, path in enumerate(fold_paths):
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()

            if bool_arr.get("f1_score", False):
                if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
                    precision = df['metrics/precision(B)']
                    recall = df['metrics/recall(B)']
                    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)  # epsilon für Division durch 0
                    temp_df = pd.DataFrame({
                        'Epoch': df[epoch_spalte],
                        'mAP': f1,
                        'Fold': f'Fold {i+1}'
                    })
                    alle_folds_df.append(temp_df)
                else:
                    print(f"Spalten für Precision oder Recall fehlen in {os.path.basename(path)}")
            else:
                if map_spalte in df.columns and epoch_spalte in df.columns:
                    temp_df = pd.DataFrame({
                        'Epoch': df[epoch_spalte],
                        'mAP': df[map_spalte],
                        'Fold': f'Fold {i+1}'
                    })
                    alle_folds_df.append(temp_df)
                else:
                    print(f"Warnung: Spalte '{map_spalte}' oder '{epoch_spalte}' fehlt in {os.path.basename(path)}")

        except FileNotFoundError:
            print(f"Fehler: Datei nicht gefunden - {path}")
        except Exception as e:
            print(f"Fehler bei Fold {i+1}: {e}")
    #print("fold creation successfull")
    return alle_folds_df, string_title

# Die Ladefunktion für einen Pfadtyp
def load_paths_for_set(set_name):
    paths = []
    for i in range(5):
        path = rf"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\cross_validation\{set_name}\fold{i}\results.csv"
        if os.path.exists(path):
            paths.append(path)
        else:
            print(f"Datei nicht gefunden: {path}")
    return paths

# Hauptfunktion: lädt für alle gültigen Sets die Pfade
def load_all_sets(set_arr):
    set_path_folds_arr = {}

    for key, value in set_arr.items():
        if value is not None:
            set_path_folds_arr[key] = load_paths_for_set(value)

    return set_path_folds_arr


def load_best_model_files(setA, setB):
    fold_paths_setA = []
    fold_paths_setB = []

    for i in range(5):
        pathA = rf"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\cross_validation\{setA}\fold{i}\results.csv"
        if os.path.exists(pathA):
            fold_paths_setA.append(pathA)
        else:
            print(f"Datei nicht gefunden: {pathA}")

        pathB = rf"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\cross_validation\{setB}\fold{i}\results.csv"
        if os.path.exists(pathB):
            fold_paths_setB.append(pathB)
        else:
            print(f"Datei nicht gefunden: {pathB}")

    return fold_paths_setA, fold_paths_setB

def create_sav_dir(bool_arr):
        
    for key, value in bool_arr.items():
        if value is True:
            comp_val = key
            break

   
    path = rf'C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\master_thesis\MA-Thesis-Latex\images\rgbir\{str(comp_val)}'
    
    try:
        os.makedirs(path)
        print(f"Ordnerstruktur '{path}' erfolgreich erstellt.")
    except FileExistsError:
        print(f"Einige oder alle Ordner in '{path}' existieren bereits.")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")


if __name__ == "__main__":
    main()