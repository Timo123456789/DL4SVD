import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def read_confusion_matrix(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith("True\\Predicted"):
            start_row = idx
            break
    cm = pd.read_csv(csv_path, skiprows=start_row)
    cm.set_index(cm.columns[0], inplace=True)
    return cm

def plot_and_save_diff(model_1, fold_1, model_2, fold_2, out_path):
    csv_path1 = rf"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\cross_validation\new_val_detect\{model_1}\fold{fold_1}\metrics_and_confusion_test.csv"
    csv_path2 = rf"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\cross_validation\new_val_detect\{model_2}\fold{fold_2}\metrics_and_confusion_test.csv"
    cm1 = read_confusion_matrix(csv_path1)
    cm2 = read_confusion_matrix(csv_path2)
    cm1_normalized = cm1.div(cm1.sum(axis=1), axis=0).fillna(0)
    cm2_normalized = cm2.div(cm2.sum(axis=1), axis=0).fillna(0)
    diff_matrix_normalized = cm1_normalized - cm2_normalized
    mask = diff_matrix_normalized == 0

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        diff_matrix_normalized.T,      # Transponiere die Matrix!
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        cbar=True,
        mask=mask.T,                  # Maske ebenfalls transponieren!
        linewidths=0.5,
        linecolor='white'
    )
    plt.title(f"Normalized Difference Matrix ({model_1}_F{fold_1} vs {model_2}_F{fold_2})")
    plt.ylabel("Predicted Label")     # Achsenbeschriftungen getauscht
    plt.xlabel("True Label")
    plt.tight_layout()
    svg_out_path = out_path.replace(".png", ".svg")
    plt.savefig(svg_out_path, format="svg", transparent=True)
    plt.close()


def plot_and_save_confusion(model, fold, out_path):
    csv_path = rf"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\cross_validation\new_val_detect\{model}\fold{fold}\metrics_and_confusion_test.csv"
    cm = read_confusion_matrix(csv_path)
    cm_normalized = cm.div(cm.sum(axis=1), axis=0).fillna(0)
    mask = cm_normalized == 0  # Nullwerte maskieren
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized.T,           # Achsen tauschen
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=True,
        mask=mask.T,               # Maske transponieren!
        linewidths=0.5,
        linecolor='white'
    )
    plt.title(f"Normalized Confusion Matrix ({model}_F{fold})")
    plt.ylabel("Predicted Label")
    plt.xlabel("True Label")
    plt.tight_layout()
    plt.savefig(out_path)
    svg_out_path = out_path.replace(".png", ".svg")
    plt.savefig(svg_out_path, format="svg", transparent=True)
    plt.close()
if __name__ == "__main__":
    output_dir = r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\master_thesis\MA-Thesis-Latex\images\confusion_matrices\Difference_Matrices"
   

    plot_and_save_confusion("RGBIR","2", r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\master_thesis\MA-Thesis-Latex\images\confusion_matrices\rgbir_F2.png" )
    plot_and_save_confusion("RGIR","3", r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\master_thesis\MA-Thesis-Latex\images\confusion_matrices\rgir_F3.png" )
    print("Test")
    # Liste der gewünschten Vergleiche (Modell1, Fold1, Modell2, Fold2, Ausgabedateiname)
    comparisons = [
        ("RGBIR", 2, "RGB", 2, "RGBIR_F2_vs_RGB_F2.png"),
        ("RGBIR", 2, "IRGB", 3, "RGBIR_F2_vs_IRGB_F3.png"),
        ("RGBIR", 2, "RIRB", 2, "RGBIR_F2_vs_RIRB_F2.png"),
        ("RGBIR", 2, "RGIR", 3, "RGBIR_F2_vs_RGIR_F3.png"),  
        ("RGBIR", 2, "GBNDVI", 2, "RGBIR_F2_vs_GBNDVI_F2.png"),
        ("RGBIR", 2, "RGBNDVI", 2, "RGBIR_F2_vs_RGBNDVI_F2.png"),
        
        
    ]
    for m1, f1, m2, f2, fname in comparisons:
       out_path = f"{output_dir}\\{fname}"
       plot_and_save_diff(m1, f1, m2, f2, out_path)
       if m1 == "RGBIR" and f1 == 2 and m2 == "RGIR" and f2 == 3:
           print("test2")
           plot_and_save_confusion(m1,f1, r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\master_thesis\MA-Thesis-Latex\images\confusion_matrices\m1.png" )
           plot_and_save_confusion(m2,f2, r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\master_thesis\MA-Thesis-Latex\images\confusion_matrices\m2.png" )
       print("saved: " + str(fname))
    print("finished")