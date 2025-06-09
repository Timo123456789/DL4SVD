import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
def main():
    # === Pfade zu deinen 10 Folds ===
    fold_paths_aab = [
        r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\obb_aab_runs\aab\orFalse_Ep500_F1/results.csv",
        r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\obb_aab_runs\aab\orFalse_Ep500_F2/results.csv",
        r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\obb_aab_runs\aab\orFalse_Ep500_F3/results.csv",
        r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\obb_aab_runs\aab\orFalse_Ep500_F4/results.csv",
        r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\obb_aab_runs\aab\orFalse_Ep500_F5/results.csv",
        r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\obb_aab_runs\aab\orFalse_Ep500_F6/results.csv",
        r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\obb_aab_runs\aab\orFalse_Ep500_F7/results.csv",
        r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\obb_aab_runs\aab\orFalse_Ep500_F8/results.csv",
        r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\obb_aab_runs\aab\orFalse_Ep500_F9/results.csv",
        r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\obb_aab_runs\aab\orFalse_Ep500_F10/results.csv",
    ]

    fold_paths_obb = [
        r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\obb_aab_runs\obb\orTrue_Ep500_F1/results.csv",
        r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\obb_aab_runs\obb\orTrue_Ep500_F2/results.csv",
        r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\obb_aab_runs\obb\orTrue_Ep500_F3/results.csv",
        r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\obb_aab_runs\obb\orTrue_Ep500_F4/results.csv",
        r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\obb_aab_runs\obb\orTrue_Ep500_F5/results.csv",
        r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\obb_aab_runs\obb\orTrue_Ep500_F6/results.csv",
        r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\obb_aab_runs\obb\orTrue_Ep500_F7/results.csv",
        r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\obb_aab_runs\obb\orTrue_Ep500_F8/results.csv",
        r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\obb_aab_runs\obb\orTrue_Ep500_F9/results.csv",
        r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\obb_aab_runs\obb\orTrue_Ep500_F10/results.csv",
    ]

    bool_arr = {
        "mAP@50-95": True,
        "mAP@50": False,
        "recall": False,
        "precision": False,
    }
    
    window_size = 20
    all_folds_aab, string_title = get_smooth_Folds(fold_paths_aab,window_size, bool_arr)
    all_folds_obb, string_title = get_smooth_Folds(fold_paths_obb,window_size, bool_arr)

   


    palette_blue = sns.color_palette("Blues", n_colors=10)
    palette_orange = sns.color_palette("Oranges", n_colors=10)

    print("test")




    if all_folds_aab:
        combined_df_aab = pd.concat(all_folds_aab, ignore_index=True)
        combined_df_aab['Type'] = 'AAB'

        combined_df_obb = pd.concat(all_folds_obb, ignore_index=True)
        combined_df_obb['Type'] = 'OBB'

        plt.figure(figsize=(14, 8))

                
        # AAB (blue)
        lines_aab = sns.lineplot(
            data=combined_df_aab,
            x='Epoch',
            y='mAP',
            hue='Fold',
            palette=palette_blue,
            linewidth=1,
            legend=False
        )

        # OBB (orange)
        lines_obb = sns.lineplot(
            data=combined_df_obb,
            x='Epoch',
            y='mAP',
            hue='Fold',
            palette=palette_orange,
            linewidth=1,
            legend=False
        )

        # Manuelle Legenden mit allen Folds
        from matplotlib.lines import Line2D

        legend_elements_aab = [Line2D([0], [0], color=palette_blue[i], lw=2, label=f'AAB Fold {i+1}') for i in range(10)]
        legend_elements_obb = [Line2D([0], [0], color=palette_orange[i], lw=2, label=f'OBB Fold {i+1}') for i in range(10)]

        legend1 = plt.legend(handles=legend_elements_aab, title='AAB Folds', loc='upper left', bbox_to_anchor=(1.01, 1))
        plt.gca().add_artist(legend1)

        legend2 = plt.legend(handles=legend_elements_obb, title='OBB Folds', loc='upper left', bbox_to_anchor=(1.01, 0.5))

        plt.title(f'{string_title} over Epochs – 10-Fold Cross Validation (window size: {window_size})')
        plt.xlabel('Epoch')
        plt.ylabel(string_title)
        plt.grid(True, linestyle=':', alpha=0.6)
        sns.despine()
        plt.tight_layout()
        plt.show()

    else:
        print("No valid data found for plotting.")


def get_smooth_Folds(fold_paths, window_size, bool_arr):
    if bool_arr["mAP@50-95"] == True:
        map_spalte = 'metrics/mAP50-95(B)'
        string_title = "mAP50-95(B)"
    elif bool_arr["mAP@50"] == True:
        map_spalte = 'metrics/mAP50(B)'
        string_title = "mAP50(B)"
    elif bool_arr["recall"] == True:
        map_spalte = 'metrics/recall(B)'
        string_title = "recall(B)"
    elif bool_arr["precision"] == True:
        map_spalte = 'metrics/precision(B)'
        string_title = "precision(B)"
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

main()