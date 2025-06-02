import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Pfade zu deinen 10 Folds ===
fold_paths = [
    r'C:\Pfad\zu\Fold1.csv',
    r'C:\Pfad\zu\Fold2.csv',
    r'C:\Pfad\zu\Fold3.csv',
    r'C:\Pfad\zu\Fold4.csv',
    r'C:\Pfad\zu\Fold5.csv',
    r'C:\Pfad\zu\Fold6.csv',
    r'C:\Pfad\zu\Fold7.csv',
    r'C:\Pfad\zu\Fold8.csv',
    r'C:\Pfad\zu\Fold9.csv',
    r'C:\Pfad\zu\Fold10.csv'
]

map_spalte = 'metrics/mAP50-95(B)'
epoch_spalte = 'epoch'
alle_folds_df = []

# === Alle mAP-Werte + Epochen pro Fold extrahieren ===
for i, path in enumerate(fold_paths):
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()

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

# === Plotten ===
if alle_folds_df:
    combined_df = pd.concat(alle_folds_df, ignore_index=True)

    plt.figure(figsize=(14, 8))
    sns.lineplot(data=combined_df, x='Epoch', y='mAP', hue='Fold', palette='tab10', linewidth=1)

    plt.title('mAP@50-95 über Epochen – 10-Fold Cross Validation')
    plt.xlabel('Epoch')
    plt.ylabel('mAP@50-95')
    plt.grid(True, linestyle=':', alpha=0.6)
    sns.despine()
    plt.tight_layout()
    plt.show()
else:
    print("Keine gültigen Daten zum Plotten gefunden.")
