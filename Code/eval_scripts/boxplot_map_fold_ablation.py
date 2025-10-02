import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.lines import Line2D  # Manuelle Legenden mit allen Folds

def main():
    """
    Main routine for loading ablation experiment results, extracting mAP@50-95 values,
    identifying best validation folds, and generating boxplots for both validation and test sets.

    - Defines channel sets and configuration flags.
    - Loads paths for validation and test results.
    - Extracts mAP@50-95 values for each fold and model.
    - Identifies the best-performing fold per model.
    - Generates boxplots for validation and test results, highlighting best folds.
    """
    all_sets = ["red", "green","blue", "ir", "ndvi"]
    set_arr = {s: s for s in all_sets}
    window_size = 20

    # Nur mAP@50-95 wird aktiviert
    bool_arr = {
        "train/box_loss": False,
        "train/cls_loss": False,
        "train/dfl_loss": False,
        "precision": True,
        "recall": False,
        "mAP@50": False,
        "mAP@50-95": False,
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
    """
    Extracts the mAP@50-95 value from a given CSV file.

    Args:
        csv_path (str): Path to the CSV file containing metrics.

    Returns:
        float or None: The extracted mAP@50-95 value, or None if extraction fails.
    """
    try:
        # Nur Zeilen 1 und 2 einlesen (Index 0 wird übersprungen)
        df = pd.read_csv(csv_path, skiprows=1, nrows=1)
        return df['metrics/mAP50-95(B)'].iloc[0]
    except Exception as e:
        print(f"Fehler beim Einlesen von {csv_path}: {e}")
        return None

def load_paths_for_set(set_name, test_bool):
    """
    Loads the file paths for all folds of a given model/channel set.

    Args:
        set_name (str): Name of the model/channel set.
        test_bool (bool): If True, loads test CSVs; otherwise, loads validation CSVs.

    Returns:
        list: List of file paths for each fold (existing files only).
    """
    paths = []
    file_name = "fold"
    for i in range(5):
        if test_bool:
            path = rf"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\cross_validation_ablation\{set_name}\{file_name}{i}\metrics_and_confusion_test.csv"
        else:
            path = rf"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\cross_validation_ablation\{set_name}\{file_name}{i}\metrics_and_confusion_val.csv"
        if os.path.exists(path):
            paths.append(path)
        else:
            print(f"Datei nicht gefunden: {path}")
    return paths

def load_all_sets(set_arr, test_bool):
    """
    Loads file paths for all channel sets and their folds.

    Args:
        set_arr (dict): Dictionary mapping set names to channel names.
        test_bool (bool): If True, loads test CSVs; otherwise, loads validation CSVs.

    Returns:
        dict: Dictionary mapping set names to lists of file paths per fold.
    """
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
    """
    Generates and displays a boxplot of mAP@50-95 values for each model/channel set across all folds.

    Args:
        set_paths_dict (dict): Dictionary mapping set names to lists of file paths per fold.
        window_size (int): Unused parameter (for compatibility).
        bool_arr (dict): Dictionary of metric flags (unused in this function).

    Displays:
        Boxplot and stripplot of mAP@50-95 values per model.
        Prints best fold and mAP@50-95 value for each model.
        Saves the plot as an SVG file.
    """
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
    df['Modell'] = df['Modell'].replace({'red': 'Red'}).replace({'green': 'Green'}).replace({'blue': 'Blue'}).replace({'ir': 'IR'}).replace({'ndvi': 'NDVI'})     # Hier ist die Änderung
    
    # Plot erstellen
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Modell', y='mAP@50-95', palette='Set2')
    sns.stripplot(data=df, x='Modell', y='mAP@50-95', color='black', alpha=0.5, jitter=False, dodge=True)

    #plt.title("mAP@50-95 per Model (for the best validation dataset on the validation data)", fontsize=14)
    plt.ylabel("mAP@0.5-0.95", fontsize=14)
    plt.xlabel("Model", fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    allsets_string = "_".join(set_paths_dict.keys()) 
    allsets_string="ablation"
    plt.savefig(r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\master_thesis\MA-Thesis-Latex\images\015Results\03ablation\best_val_on_val.svg", format="svg", transparent=True)
    
    print("val on val")
    # Für jedes Modell den besten mAP@50-95 Wert ausgeben
    for modell in df['Modell'].unique():
        df_modell = df[df['Modell'] == modell]
        best_idx = df_modell['mAP@50-95'].idxmax()
        best_row = df_modell.loc[best_idx]
        print(f"Best for '{modell}': Fold={best_row['Fold']}, mAP@50-95={best_row['mAP@50-95']:.4f}")
    plt.tight_layout()
    plt.show()

    


def create_boxplot_from_sets_red_dot(set_paths_dict, window_size, bool_arr, best_fold_indices=None):
    """
    Generates and displays a boxplot of mAP@50-95 values for each model/channel set,
    marking the best validation fold for each model with a red dot.

    Args:
        set_paths_dict (dict): Dictionary mapping set names to lists of file paths per fold.
        window_size (int): Unused parameter (for compatibility).
        bool_arr (dict): Dictionary of metric flags (unused in this function).
        best_fold_indices (dict, optional): Dictionary mapping model names to best fold indices.

    Displays:
        Boxplot and stripplot of mAP@50-95 values per model.
        Highlights best validation fold with a red dot.
        Prints best fold and mAP@50-95 value for each model.
        Saves the plot as an SVG file.
    """
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
    df['Modell'] = df['Modell'].replace({'red': 'Red'}).replace({'green': 'Green'}).replace({'blue': 'Blue'}).replace({'ir': 'IR'}).replace({'ndvi': 'NDVI'})
   

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Modell', y='mAP@50-95', palette='Set2')
    sns.stripplot(data=df, x='Modell', y='mAP@50-95', color='black', alpha=0.5, jitter=False, dodge=True)

    # Rote Punkte für beste Folds markieren
    if best_fold_indices is not None:
        modell_order = df['Modell'].unique()
        for modell, best_fold in best_fold_indices.items():
            # Filtere den mAP Wert für das Modell und den besten Fold
            val = df[(df['Modell'] == modell) & (df['Fold'] == best_fold)]['mAP@50-95']
            
            if not val.empty:
                x_pos = list(modell_order).index(modell)
                y_val = val.values[0]
                plt.scatter(x_pos, y_val, color='red', s=15, edgecolor='red', zorder=10)

    #plt.title("mAP@50-95 per Model (across 5 folds) -  best validation model's performance on the test fold", fontsize=10)

    
    plt.ylabel("mAP@0.5-0.95", fontsize=14)
    plt.xlabel("Model", fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    # Hinweis zum roten Punkt im Plot hinzufügen
    #plt.text(0.95, 0.02, "The red dot indicates the fold of the best validation model on the test data.",
    #        horizontalalignment='right', verticalalignment='bottom',
    #        transform=plt.gca().transAxes,
    #        fontsize=5, color='red', alpha=0.7)

    # Legende: 'Best Validation Fold' nur einmal anzeigen
    if best_fold_indices:
        handles, labels = plt.gca().get_legend_handles_labels()
        if 'Best Validation Fold' in labels:
            plt.legend(handles=[handles[labels.index('Best Validation Fold')]], labels=['Best Validation Fold'])

    allsets_string = "_".join(set_paths_dict.keys()) 
    allsets_string="ablation"
    plt.savefig(r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\master_thesis\MA-Thesis-Latex\images\015Results\03ablation\best_val_on_test.svg", format="svg", transparent=True)

# Für jedes Modell den besten mAP@50-95 Wert ausgeben
    print("val on test")
    for modell in df['Modell'].unique():
        df_modell = df[df['Modell'] == modell]
        best_idx = df_modell['mAP@50-95'].idxmax()
        best_row = df_modell.loc[best_idx]
        print(f"Best for '{modell}': Fold={best_row['Fold']}, mAP@50-95={best_row['mAP@50-95']:.4f}")


    plt.tight_layout()
    plt.show()

    

def get_top_val_folds(set_paths_dict, top_n=2):
    """
    Returns the indices of the top-N validation folds per model based on mAP@50-95.

    Args:
        set_paths_dict (dict): Dictionary mapping set names to lists of file paths per fold.
        top_n (int): Number of top folds to return per model.

    Returns:
        dict: Dictionary mapping set names to lists of top-N fold indices.
    """
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
    """
    Creates a boxplot of test mAP@50-95 values from the top-N validation folds,
    highlighting the best-performing fold in red.

    Args:
        val_paths_dict (dict): Dictionary mapping set names to validation file paths per fold.
        test_paths_dict (dict): Dictionary mapping set names to test file paths per fold.
        top_n (int): Number of top validation folds to consider per model.

    Displays:
        Boxplot and scatterplot of test mAP@50-95 values for top validation folds.
        Highlights the best-performing fold in red.
        Saves the plot as an SVG file.
    """
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
    plt.ylabel("mAP@50-95 (Test)", fontsize=14)
    plt.xlabel("Model", fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    allsets_string = "_".join(test_paths_dict.keys()) 
    plt.savefig(allsets_string + "_best_val_on_test.svg", format="svg", transparent=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()