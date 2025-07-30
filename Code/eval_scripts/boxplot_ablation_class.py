import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.lines import Line2D  # Manuelle Legenden mit allen Folds
import glob
import re

def main():
    all_sets = ["r", "g","b", "ir", "ndvi"]
    set_arr = {s: s for s in all_sets}

    bool_value= {
        "boxp": False,
        "Recall": False,
        "map50": False,
        "map50-95": True,
    }

    # Pfad zum Hauptverzeichnis, z.B. './models'
    main_dir = r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\output\abl"
    out_path = r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\master_thesis\MA-Thesis-Latex\images\ablation\boxplot"
    # Alle .dat-Dateien finden (rekursiv)


    extracted_values = get_file_Values(main_dir)
    print(extracted_values[0])
    count_boxp_values_per_channel_and_class(extracted_values, all_sets)
    #create_bp(extracted_values, all_sets, "Car")
    create_bp_all_classes(extracted_values, all_sets, out_path, bool_value)


def count_boxp_values_per_channel_and_class(extracted_values, all_sets):
    # Alle Klassen sammeln
    all_class_names = set()
    for file_dict in extracted_values:
        all_class_names.update(file_dict["classes"].keys())
    all_class_names = sorted(all_class_names)

    for selected_class in all_class_names:
        for channel in all_sets:
            count = 0
            for file_dict in extracted_values:
                channel_found = get_channel_from_path(file_dict["file"], all_sets)
                if channel_found == channel and selected_class in file_dict["classes"]:
                    class_dict = file_dict["classes"][selected_class]
                    try:
                        val = float(class_dict.get("Box(P", "nan"))
                        if not pd.isna(val):
                            count += 1
                    except Exception:
                        continue
            print(f"Anzahl aller Box(P) Werte für Klasse '{selected_class}' im Kanal '{channel}': {count}")


def create_bp(extracted_values, all_sets, selected_class):
    data = []
    for file_dict in extracted_values:
        # Kanal aus Pfad extrahieren (z.B. .../r/fold0.dat → r)
        for channel in all_sets:
            if channel in os.path.basename(os.path.dirname(file_dict["file"])).lower():
                # Nur die Klasse "Car" nehmen
                
                class_dict = file_dict["classes"][selected_class]
                try:
                    box_p = float(class_dict.get("Box(P", class_dict.get("Box(P)", "nan")))
                    data.append({"Channel": channel, "Box(P)": box_p})
                except Exception:
                    continue

    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Channel", y="Box(P)", palette="Set2")
    plt.title(f"Boxplot of Box(P) values for class {selected_class} per Channel")
    plt.ylabel("Box(P)")
    plt.xlabel("Channel")
    plt.tight_layout()
    plt.show()


def create_bp_all_classes(extracted_values, all_sets, out_path, bool_values):
    # Alle Klassen sammeln
    all_class_names = set()
    for file_dict in extracted_values:
        all_class_names.update(file_dict["classes"].keys())
    all_class_names = sorted(all_class_names)

    # Nur die Metriken plotten, die in bool_values auf True stehen
    metrics = {
        "boxp": {"key": "Box(P", "label": "Box(P)", "ylabel": "Box(P)"},
        "Recall": {"key": "R", "label": "Recall", "ylabel": "Recall"},
        "map50": {"key": "mAP50", "label": "mAP50", "ylabel": "mAP50"},
        "map50-95": {"key": "mAP50-95)", "label": "mAP50-95", "ylabel": "mAP50-95"},
    }
    # Nur die gewünschten Metriken auswählen
    selected_metrics = {k: v for k, v in metrics.items() if bool_values.get(k, False)}

    for selected_class in all_class_names:
        for metric_name, metric_info in selected_metrics.items():
            data = []
            for file_dict in extracted_values:
                channel_found = get_channel_from_path(file_dict["file"], all_sets)
                if channel_found and selected_class in file_dict["classes"]:
                    class_dict = file_dict["classes"][selected_class]
                    try:
                        value = float(class_dict.get(metric_info["key"], "nan"))
                        data.append({"Channel": channel_found, metric_info["label"]: value})
                    except Exception:
                        continue
            if data:
                df = pd.DataFrame(data)
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=df, x="Channel", y=metric_info["label"], palette="Set2", order=all_sets)
                plt.title(f"Boxplot of {metric_info['label']} values for class '{selected_class}' per Channel")
                plt.ylabel(metric_info["ylabel"])
                plt.xlabel("Channel")
                svg_out_path = os.path.join(out_path, "svg", metric_info['label'], f"{selected_class}_{metric_info['label']}.svg")
                png_out_path = os.path.join(out_path, "png", metric_info['label'], f"{selected_class}_{metric_info['label']}.png")
                os.makedirs(os.path.dirname(svg_out_path), exist_ok=True)
                os.makedirs(os.path.dirname(png_out_path), exist_ok=True)
                plt.savefig(svg_out_path, format="svg", transparent=True)
                plt.savefig(png_out_path, transparent=False)
                plt.tight_layout()
                #plt.show()


def get_file_Values(main_dir):
    #dat_files = glob.glob(os.path.join(main_dir, '**', '*.dat'), recursive=True)
    dat_files = glob.glob(os.path.join(main_dir, '*', 'fold*.dat'))
    extracted_values = []
    for dat_file in dat_files:
        with open(dat_file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
            last_29 = lines[-29:]
            eleven_lines = last_29[:11]
            # Spaltennamen aus der ersten Zeile extrahieren
            header_line = eleven_lines[0]
            if ':' in header_line:
                header_line = header_line.split(':', 1)[0]
            columns = header_line.strip().split()
            file_dict = {"file": dat_file, "classes": {}}
            # Jede Klasse als Dictionary speichern
            for line in eleven_lines[1:]:
                parts = line.strip().split()
                # Sonderfall: "Camping Car" hat ein Leerzeichen im Namen
                if len(parts) == len(columns):
                    class_name = parts[0]
                elif len(parts) == len(columns) + 1 and parts[0] == "Camping" and parts[1] == "Car":
                    class_name = "Camping_Car"
                    # Die Werte für die Spalten sind ab Index 2
                    parts = [class_name] + parts[2:]
                else:
                    continue
                class_dict = {col: val for col, val in zip(columns, parts)}
                file_dict["classes"][class_name] = class_dict
            print("-"*10)
            #print(file_dict)
            print(f"Kanal: {get_channel_from_path(dat_file, ['r','g','b','ir','ndvi'])}, Datei: {dat_file}")
        extracted_values.append(file_dict)
    return extracted_values

def get_channel_from_path(filepath, all_sets):
    # Extrahiere den Kanal aus dem Pfad (z.B. .../b/fold0.dat → b)
    parts = os.path.normpath(filepath).split(os.sep)
    for part in parts:
        if part in all_sets:
            return part
    return None
            
if __name__ == "__main__":
    main()