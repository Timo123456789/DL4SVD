import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.lines import Line2D  # Manuelle Legenden mit allen Folds

def main():
    all_sets = ["aab", "obb","aab_old"]
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
        "tail": None,
    }

    set_paths_dict = load_all_sets(set_arr, test_bool=False)
    create_boxplot_from_sets(set_paths_dict, window_size, bool_arr)

    test_paths_dict = set_paths_dict

# Lade erneut VAL-Daten (für Auswahl der Top-Folds)
    val_paths_dict = load_all_sets(set_arr, test_bool=False)

    # Visualisiere Testwerte für Top-Val-Folds
    #create_boxplot_top_folds(val_paths_dict, test_paths_dict, top_n=5)

    print("Start")
    map_values_val = []  # Liste initialisieren

    for set_name, paths in set_paths_dict.items():
        print(f"\n{set_name}:")
        temp = []
        for i, path in enumerate(paths):
            map_val = extract_map50_95(path)
            if map_val is not None:
                print(f"  Fold {i} Val: mAP50-95 = {map_val:.4f}")
                temp.append([i, map_val])
            else:
                print(f"  Fold {i}: Fehler beim Einlesen.")

        temp_dict = {
            "model": set_name,
            "values": temp
        }

        map_values_val.append(temp_dict)
    best_fold_indices = {}

    for entry in map_values_val:
        model = entry['model']
        values = entry['values']
        if values:
            best_fold = max(values, key=lambda x: x[1])
            best_fold_indices[model] = best_fold[0]
    print(map_values_val)
    set_paths_dict = load_all_sets(set_arr, test_bool=True)
    create_boxplot_from_sets_red_dot(set_paths_dict, window_size, bool_arr, best_fold_indices)

    print("_________________________________________")

    for set_name, paths in set_paths_dict.items():
        print(f"\n{set_name}:")
        for i, path in enumerate(paths):
            map_val = extract_map50_95(path)
            if map_val is not None:
                print(f"  Fold {i} Test: mAP50-95 = {map_val:.4f}")
            else:
                print(f"  Fold {i}: Fehler beim Einlesen.")

    print("finished")

# Funktion zum Einlesen von mAP50-95 aus CSV
def extract_map50_95(csv_path):
    try:
        # Nur Zeilen 1 und 2 einlesen (Index 0 wird übersprungen)
        df = pd.read_csv(csv_path, skiprows=1, nrows=1)
        return df['metrics/mAP50-95(B)'].iloc[0]
    except Exception as e:
        print(f"Fehler beim Einlesen von {csv_path}: {e}")
        return None

def load_paths_for_set(set_name, test_bool):
    paths = []
    if set_name == "obb":
        file_name= "orTrue_Ep500_F"
    else:
        file_name= "orFalse_Ep500_F"
    for i in range(5):
        if test_bool:
            path = rf"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\obb_aab_runs\{set_name}\{file_name}{i}\metrics_and_confusion_test.csv"
        else:
            path = rf"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\obb_aab_runs\{set_name}\{file_name}{i}\metrics_and_confusion_val.csv"
        if os.path.exists(path):
            paths.append(path)
        else:
            print(f"Datei nicht gefunden: {path}")
    return paths

def load_all_sets(set_arr, test_bool):
    set_path_folds_arr = {}
    for key, value in set_arr.items():
        if value is not None:
            set_path_folds_arr[key] = load_paths_for_set(value, test_bool)
    return set_path_folds_arr

# Beispielhafte Visualisierungsfunktion (Platzhalter)
# def create_boxplot_from_sets(set_paths_dict, window_size, bool_arr):
#     for set_name, paths in set_paths_dict.items():
#         map_values = []
#         for path in paths:
#             map_val = extract_map50_95(path)
#             if map_val is not None:
#                 map_values.append(map_val)
#         print(f"{set_name}: mAP50-95 Werte: {map_values}")
#         # Optional: hier könntest du Boxplots aus map_values erstellen

def create_boxplot_from_sets(set_paths_dict, window_size, bool_arr):
    all_data = []

    for set_name, paths in set_paths_dict.items():
        for fold_idx, path in enumerate(paths):
            map_val = extract_map50_95(path)
            if map_val is not None:
                all_data.append({
                    'Modell': set_name,
                    'mAP@50-95': map_val,
                    'Fold': fold_idx
                })

    if not all_data:
        print("Keine gültigen Daten zum Plotten gefunden.")
        return

    df = pd.DataFrame(all_data)
    df['Modell'] = df['Modell'].replace({'aab_old': 'ABB in OBB'})  
    df['Modell'] = df['Modell'].replace({'aab': 'ABB'})
    df['Modell'] = df['Modell'].replace({'obb': 'OBB'})
    
    print("val on val")
    print(df)

    # Plot erstellen
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Modell', y='mAP@50-95', palette={"OBB": "orange", "ABB": "steelblue", "ABB in OBB": "green"})
    sns.stripplot(data=df, x='Modell', y='mAP@50-95', color='black', alpha=0.5, jitter=False, dodge=True)

    #plt.title("mAP@50-95 per Model (for the best validation dataset on the validation data)", fontsize=14)
    plt.ylabel("mAP@0.5-0.95", fontsize=14)
    plt.xlabel("Model", fontsize=14)
    plt.xticks(rotation=0, fontsize=12)  # Ticklabels an der X-Achse
    plt.yticks(fontsize=12)              # Ticklabels an der Y-Achse
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    allsets_string = "_".join(set_paths_dict.keys())
    allsets_string = "abb_obb"
    plt.savefig(r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\master_thesis\MA-Thesis-Latex\images\015Results\01abb_vs_obb\abb_obb_best_val_on_val.svg", format="svg", transparent=True)
    plt.tight_layout()
    plt.show()


def create_boxplot_from_sets_red_dot(set_paths_dict, window_size, bool_arr, best_fold_indices=None):
    all_data = []

    for set_name, paths in set_paths_dict.items():
        for fold_idx, path in enumerate(paths):
            map_val = extract_map50_95(path)
            if map_val is not None:
                all_data.append({
                    'Modell': set_name,
                    'mAP@50-95': map_val,
                    'Fold': fold_idx
                })

    if not all_data:
        print("Keine gültigen Daten zum Plotten gefunden.")
        return

    df = pd.DataFrame(all_data)

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=df, x='Modell', y='mAP@50-95', palette={"obb": "orange", "aab": "steelblue", "aab_old": "green"})
    sns.stripplot(data=df, x='Modell', y='mAP@50-95', color='black', alpha=0.5, jitter=False, dodge=True)

    # Rote Punkte für beste Folds markieren
    if best_fold_indices is not None:
        modell_order = df['Modell'].unique()
        for modell, best_fold in best_fold_indices.items():
            val = df[(df['Modell'] == modell) & (df['Fold'] == best_fold)]['mAP@50-95']
            if not val.empty:
                x_pos = list(modell_order).index(modell)
                y_val = val.values[0]
                plt.scatter(
                    x_pos, y_val, color='red', s=15, edgecolor='red',
                    zorder=10, label='Best Validation Fold' if modell == list(best_fold_indices.keys())[0] else ""
                )

    # NACH dem Zeichnen Labels ändern
    new_labels = [
    lbl.get_text().replace("aab_old", "ABB in OBB").replace("aab", "ABB").replace("obb", "OBB")
    for lbl in ax.get_xticklabels()
    ]
    ax.set_xticklabels(new_labels)

    ax.set_xlabel("Model", fontsize=14)
    ax.set_ylabel("mAP@0.5-0.95", fontsize=14)
    ax.set_xticklabels(new_labels, fontsize=12)

    plt.ylabel("mAP@0.5-0.95")
    plt.xlabel("Bounding Box Type")
    plt.xticks(rotation=0)

    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    # Hinweis zum roten Punkt im Plot hinzufügen
    #plt.text(0.95, 0.02, "The red dot indicates the fold of the best validation model on the test data.",
    #        horizontalalignment='right', verticalalignment='bottom',
    #        transform=plt.gca().transAxes,
    #        fontsize=5, color='red', alpha=0.7)

    # Legende: 'Best Validation Fold' nur einmal anzeigen
    # if best_fold_indices:
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     if 'Best Validation Fold' in labels:
    #         plt.legend(handles=[handles[labels.index('Best Validation Fold')]], labels=['Best Validation Fold'])
    print("best_fold_indices val on test")
    print(best_fold_indices)

    allsets_string = "_".join(set_paths_dict.keys()) 
    allsets_string = "aab_obb"
    plt.savefig(r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\master_thesis\MA-Thesis-Latex\images\015Results\01abb_vs_obb\aab_obb_best_val_on_test.svg", format="svg", transparent=True)
    plt.tight_layout()
    plt.show()

def get_top_val_folds(set_paths_dict, top_n=2):
    """Gibt die Indizes der Top-N Validierungsfolds pro Modell zurück."""
    top_folds = {}

    for set_name, paths in set_paths_dict.items():
        fold_scores = []
        for fold_idx, path in enumerate(paths):
            map_val = extract_map50_95(path)
            if map_val is not None:
                fold_scores.append((fold_idx, map_val))
        # Sortiere nach mAP und nimm die Top-N
        fold_scores.sort(key=lambda x: x[1], reverse=True)
        top_folds[set_name] = [idx for idx, val in fold_scores[:top_n]]

    return top_folds

def create_boxplot_top_folds(val_paths_dict, test_paths_dict, top_n=1):
    """Creates a bar plot of the test mAP@50-95 values from the top-N validation folds, with the best one highlighted in red."""
    top_folds = get_top_val_folds(val_paths_dict, top_n=top_n)
    all_data = []

    best_point = None  # To highlight the best point in red

    for set_name, test_paths in test_paths_dict.items():
        if set_name not in top_folds:
            continue
        for fold_idx in top_folds[set_name]:
            if fold_idx >= len(test_paths):
                continue
            map_val = extract_map50_95(test_paths[fold_idx])
            if map_val is not None:
                all_data.append({
                    'Model': set_name,
                    'mAP@50-95 (Test)': map_val,
                    'Fold': fold_idx
                })

    if not all_data:
        print("No data found for top folds.")
        return

    df = pd.DataFrame(all_data)

    # Identify best-performing test fold
    best_idx = df['mAP@50-95 (Test)'].idxmax()
    best_point = df.loc[best_idx]

    # Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Model', y='mAP@50-95 (Test)', palette='Set2')

    # Plot all points, highlight best in red
    for i, row in df.iterrows():
        color = 'red' if i == best_idx else 'black'
        plt.scatter(
            x=row['Model'],
            y=row['mAP@50-95 (Test)'],
            color=color,
            edgecolor='white',
            zorder=5,
            s=80,
            label=' Best Validation Fold' if color == 'red' else ""
        )

    # Add legend for best point
    handles, labels = plt.gca().get_legend_handles_labels()
    if ' Best Validation Fold' in labels:
        legend_element = Line2D([0], [0], marker='o', color='w', label=' Best Validation Fold',
                                markerfacecolor='red', markersize=10)
        plt.legend(handles=[legend_element])

    plt.title(f"Test mAP@50-95 of Top-{top_n} Validation Folds per Model")
    plt.ylabel("mAP@50-95 (Test)")
    plt.xlabel("Model")
    plt.xticks(rotation=0)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()