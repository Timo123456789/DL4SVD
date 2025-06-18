import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
    # Manuelle Legenden mit allen Folds
from matplotlib.lines import Line2D
def main():
    setA = r"rgbndvi"
    setB = r"rgbir"
    window_size = 20

    
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
            "tail":None,
            }
    

    fold_paths_setA, fold_paths_setB = load_files(setA, setB)

    create_graphs(fold_paths_setA,fold_paths_setB,setA,setB,window_size, bool_arr)


   

def create_graphs(fold_paths_setA,fold_paths_setB,setA,setB,window_size, bool_arr):

    
    all_folds_setA, string_title = get_smooth_Folds(fold_paths_setA,window_size, bool_arr, setA)
    all_folds_setB, string_title = get_smooth_Folds(fold_paths_setB,window_size, bool_arr, setB)

  
    
    number_folds_setA = len(fold_paths_setA)
    number_folds_setB = len(fold_paths_setB)
    print("Number folds "+ setA +": "  + str(number_folds_setA) + " Number folds  " + setB +": " + str(number_folds_setB))
    palette_blue = sns.color_palette("Blues", n_colors=number_folds_setA)
    palette_orange = sns.color_palette("Oranges", n_colors=number_folds_setB)







    combined_df_setA = pd.concat(all_folds_setA, ignore_index=True)
    combined_df_setA['Type'] = 'setA'

    combined_df_setB = pd.concat(all_folds_setB, ignore_index=True)
    combined_df_setB['Type'] = 'setB'

    if bool_arr['tail'] is not None:
        combined_df_setA = combined_df_setA.groupby('Fold').tail(bool_arr['tail'])
        combined_df_setB = combined_df_setB.groupby('Fold').tail(bool_arr['tail'])

    plt.figure(figsize=(14, 8))

    lines_setA = sns.lineplot(
        data=combined_df_setA,
        x='Epoch',
        y='mAP',
        hue='Fold',
        palette=palette_blue,
        linewidth=1,
        alpha=0.8,
        legend=False
    )

    lines_setB = sns.lineplot(
        data=combined_df_setB,
        x='Epoch',
        y='mAP',
        hue='Fold',
        palette=palette_orange,
        linewidth=2,
        alpha=0.6,
        legend=False
    )



    legend_elements_setA = [Line2D([0], [0], color=palette_blue[i], lw=2, label=f'Fold {i+1}') for i in range(number_folds_setA)]
    legend_elements_setB = [Line2D([0], [0], color=palette_orange[i], lw=2, label=f'Fold {i+1}') for i in range(number_folds_setB)]

    legend1 = plt.legend(handles=legend_elements_setA, title=setA, loc='upper left', bbox_to_anchor=(1.01, 1))
    plt.gca().add_artist(legend1)

    legend2 = plt.legend(handles=legend_elements_setB, title=setB, loc='upper left', bbox_to_anchor=(1.01, 0.5))
    

    plt.title(f'{string_title} at {setA} vs. {setB} over Epochs – 5-Fold Cross Validation (window size: {window_size})')
    plt.xlabel('Epoch')
    plt.ylabel(string_title)
    plt.grid(True, linestyle=':', alpha=0.6)
    sns.despine()
    plt.tight_layout()
    plt.show()


def get_smooth_Folds(fold_paths, window_size, bool_arr, set_string):

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
    else:
        print("Error bei Spaltenbezeichnung")
   
    epoch_spalte = 'epoch'
    alle_folds_df = []
     # === Alle mAP-Werte + Epochen pro Fold extrahieren ===
    for i, path in enumerate(fold_paths):
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()

            if map_spalte in df.columns and epoch_spalte in df.columns:
                smoothed_map = df[map_spalte].rolling(window=window_size, min_periods=1).mean()
                temp_df = pd.DataFrame({
                    'Epoch': df[epoch_spalte],
                    'mAP': smoothed_map,
                    'Fold': f'Fold {i+1}'
                })
                alle_folds_df.append(temp_df)
            else:
                print(f"Warnung: Spalte '{map_spalte}' oder '{epoch_spalte}' fehlt in {os.path.basename(path)}")

        except FileNotFoundError:
            print(f"Fehler: Datei nicht gefunden - {path}")
        except Exception as e:
            print(f"Fehler bei Fold {i+1}: {e}")
    print("fold creation successfull")
    return alle_folds_df, string_title

def load_files(setA, setB):
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



main()