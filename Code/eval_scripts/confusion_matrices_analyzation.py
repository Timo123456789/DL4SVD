import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def read_confusion_matrix(csv_path):
    """
    Reads a confusion matrix from a CSV file, skipping header rows until the matrix starts.

    Args:
        csv_path (str): Path to the CSV file containing the confusion matrix.

    Returns:
        pd.DataFrame: Confusion matrix as a pandas DataFrame with proper row and column labels.
    """
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
    """
    Plots and saves the normalized difference between two confusion matrices from different models/folds,
    masking values with absolute difference <= 0.05 and formatting class labels.

    Args:
        model_1 (str): Name of the first model.
        fold_1 (int or str): Fold number for the first model.
        model_2 (str): Name of the second model.
        fold_2 (int or str): Fold number for the second model.
        out_path (str): Output path for saving the SVG image.

    Saves:
        A heatmap of the normalized difference matrix as an SVG file, with small differences masked.
    """
    csv_path1 = rf"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\cross_validation\new_val_detect\{model_1}\fold{fold_1}\metrics_and_confusion_test.csv"
    csv_path2 = rf"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\cross_validation\new_val_detect\{model_2}\fold{fold_2}\metrics_and_confusion_test.csv"

    cm1 = read_confusion_matrix(csv_path1)
    cm2 = read_confusion_matrix(csv_path2)

    cm1_normalized = cm1.div(cm1.sum(axis=1), axis=0).fillna(0)
    cm2_normalized = cm2.div(cm2.sum(axis=1), axis=0).fillna(0)

    diff_matrix_normalized = cm1_normalized - cm2_normalized

    # Maske für Werte im Bereich [-0.05, 0.05]
    mask = diff_matrix_normalized.abs() <= 0.05

    # Klassenlabels formatieren
    class_labels = [cls.title() for cls in cm1.index]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        diff_matrix_normalized.T,      # Transponiere die Matrix!
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        cbar=True,
        mask=mask.T,                   # Maske ebenfalls transponieren!
        linewidths=0.5,
        linecolor='white',
        annot_kws={"size": 16},
        xticklabels=class_labels,   # neue Labels für X
        yticklabels=class_labels    # neue Labels für Y
    )

    plt.ylabel("Predicted Label", fontsize=12)  # Achsenbeschriftungen getauscht
    plt.xlabel("True Label", fontsize=12)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    svg_out_path = out_path.replace(".png", ".svg")
    plt.savefig(svg_out_path, format="svg", transparent=True)
    plt.close()


def plot_and_save_diff_old(model_1, fold_1, model_2, fold_2, out_path):
    """
    Plots and saves the normalized difference between two confusion matrices from different models/folds,
    masking only zero differences.

    Args:
        model_1 (str): Name of the first model.
        fold_1 (int or str): Fold number for the first model.
        model_2 (str): Name of the second model.
        fold_2 (int or str): Fold number for the second model.
        out_path (str): Output path for saving the SVG image.

    Saves:
        A heatmap of the normalized difference matrix as an SVG file, with zero differences masked.
    """
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
        linecolor='white',
        annot_kws={"size": 16},
        
    )
    #plt.title(f"Normalized Difference Matrix ({model_1}_F{fold_1} vs {model_2}_F{fold_2})")
    plt.ylabel("Predicted Label",fontsize=12)     # Achsenbeschriftungen getauscht
    plt.xlabel("True Label",fontsize=12)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    svg_out_path = out_path.replace(".png", ".svg")
    plt.savefig(svg_out_path, format="svg", transparent=True)
    plt.close()


def plot_and_save_confusion(model, fold, out_path):
    """
    Plots and saves the normalized confusion matrix for a given model and fold,
    masking zero values.

    Args:
        model (str): Name of the model.
        fold (int or str): Fold number.
        out_path (str): Output path for saving the SVG image.

    Saves:
        The normalized confusion matrix as an SVG file, with zero values masked.
    """
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
        linecolor='white',
        annot_kws={"size": 14}
    )
    #plt.title(f"Normalized Confusion Matrix ({model}_F{fold})")
    plt.ylabel("Predicted Label",fontsize=16)
    plt.xlabel("True Label",fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    #plt.savefig(out_path)
    svg_out_path = out_path.replace(".png", ".svg")
    plt.savefig(svg_out_path, format="svg", transparent=True)
    plt.close()

def plot_and_save_confusion_new(model, fold, out_path):
    """
    Plots and saves the normalized confusion matrix for a given model and fold,
    masking values with absolute value <= 0.05 and formatting class labels.

    Args:
        model (str): Name of the model.
        fold (int or str): Fold number.
        out_path (str): Output path for saving the SVG image.

    Saves:
        The normalized confusion matrix as an SVG file, with small values masked and formatted labels.
    """
    csv_path = rf"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\cross_validation\new_val_detect\{model}\fold{fold}\metrics_and_confusion_test.csv"
    cm = read_confusion_matrix(csv_path)
    cm_normalized = cm.div(cm.sum(axis=1), axis=0).fillna(0)

    # Maske für Werte im Bereich [-0.05, 0.05]
    mask = cm_normalized.abs() <= 0.05

    # Klassenlabels formatieren
    class_labels = [cls.title() for cls in cm.index]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized.T,           # Achsen tauschen
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=True,
        mask=mask.T,               # Maske transponieren!
        linewidths=0.5,
        linecolor='white',
        annot_kws={"size": 14},
        xticklabels=class_labels,   # neue Labels für X
        yticklabels=class_labels    # neue Labels für Y
    )
    #plt.title(f"Normalized Confusion Matrix ({model}_F{fold})")
    plt.ylabel("Predicted Label", fontsize=16)
    plt.xlabel("True Label", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    svg_out_path = out_path.replace(".png", ".svg")
    plt.savefig(svg_out_path, format="svg", transparent=True)
    plt.close()


if __name__ == "__main__":
    output_dir = r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\master_thesis\MA-Thesis-Latex\images\015Results\02perm_exp\confusion_matrices\Difference_Matrices"
   

    plot_and_save_confusion_new("RGBIR","2", r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\master_thesis\MA-Thesis-Latex\images\015Results\02perm_exp\confusion_matrices\rgbir_F2.png" )
    plot_and_save_confusion_new("RGIR","3", r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\master_thesis\MA-Thesis-Latex\images\015Results\02perm_exp\confusion_matrices\rgir_F3.png" )
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
           plot_and_save_confusion(m1,f1, r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\master_thesis\MA-Thesis-Latex\images\015Results\02perm_exp\confusion_matrices\m1.png" )
           plot_and_save_confusion(m2,f2, r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\master_thesis\MA-Thesis-Latex\images\015Results\02perm_exp\confusion_matrices\m2.png" )
       print("saved: " + str(fname))
    print("finished")