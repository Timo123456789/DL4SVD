import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path_obb_false= rf"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\Ep1500_pat150_obb_false\results.csv"
path_obb_true= rf"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\Ep1500_pat150_obb_true\results.csv"

path_dota_rgb= rf"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\DOTA_RGB_VEDAI\results.csv"
path_dota_rgbir = rf"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\DOTA_RGBIR\results.csv"
path_dota_irgb= rf"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\DOTA_irgb\results.csv"
path_dota_rirb = rf"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\DOTA_rirb\results.csv"
path_dota_rgir = rf"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\DOTA_RGIR\results.csv"

try:
    df_obb_false = pd.read_csv(path_obb_false)
    df_obb_true = pd.read_csv(path_obb_true)
except FileNotFoundError:
    print("Fehler: Eine Datei wurde nicht gefunden.")
    exit()



df_obb_false.columns = df_obb_false.columns.str.strip()

# 2. Relevante Spalten auswählen
epochen_spalte = 'epoch'
map_spalte = 'metrics/mAP50-95(B)'

# if epochen_spalte not in df_obb_false.columns or map_spalte not in df_obb_false.columns or \
#    epochen_spalte not in df_obb_true.columns or map_spalte not in df_obb_true.columns:
#     print(f"Fehler: Eine der benötigten Spalten ('{epochen_spalte}' oder '{map_spalte}') fehlt in einer der Dateien.")
#     exit()

epochen1 = df_obb_false[epochen_spalte]
map_werte1 = df_obb_false[map_spalte]
epochen2 = df_obb_true[epochen_spalte]
map_werte2 = df_obb_true[map_spalte]

# 3. Daten für seaborn vorbereiten (optional, aber oft hilfreich für komplexere Plots)
# Hier erstellen wir DataFrames, die die Daten und eine Kennzeichnung der Datei enthalten
data1 = pd.DataFrame({epochen_spalte: epochen1, map_spalte: map_werte1, 'Modell': 'Axis Aligned Bounding Boxes'})
data2 = pd.DataFrame({epochen_spalte: epochen2, map_spalte: map_werte2, 'Modell': 'Oriented Bounding Boxen'})

# Die beiden DataFrames zusammenführen
vergleich_df = pd.concat([data1, data2], ignore_index=True)

# 4. Visualisierung mit seaborn
plt.figure(figsize=(12, 7))
sns.lineplot(x=epochen_spalte, y=map_spalte, hue='Modell', data=vergleich_df, marker='', linewidth=0.8)

plt.xlabel('Epoch')
plt.ylabel('mAP@50-95')
plt.title('Comparison by mAP@50-95 between Axis Aligned Bounding Boxes and Oriented Bounding Boxes')
plt.grid(True, linestyle=':', alpha=0.7)
sns.despine() # Entfernt obere und rechte Achsenlinien
plt.tight_layout()
plt.show()


# Fenstergröße für den gleitenden Durchschnitt
window_size = 100

dataframes = {}
try:
    dataframes['RGB'] = pd.read_csv(path_dota_rgb)
    dataframes['RGBIR'] = pd.read_csv(path_dota_rgbir)
    dataframes['IRGB'] = pd.read_csv(path_dota_irgb)
    dataframes['RIRB'] = pd.read_csv(path_dota_rirb)
    dataframes['RGIR'] = pd.read_csv(path_dota_rgir)
except FileNotFoundError:
    print("Fehler: Eine oder mehrere CSV-Dateien wurden nicht gefunden.")
    exit()

# Daten für seaborn vorbereiten
# Daten für seaborn vorbereiten (mit geglätteten Werten)
vergleich_df_list = []
for name, df in dataframes.items():
    if epochen_spalte in df.columns and map_spalte in df.columns:
        # Gleitenden Durchschnitt berechnen
        smoothed_map = df[map_spalte].rolling(window=window_size, min_periods=1).mean()

        # Originaldaten
        #temp_df_orig = pd.DataFrame({epochen_spalte: df[epochen_spalte],
        #                             map_spalte: df[map_spalte],
        #                             'Modell': f'Dota {name} (Original)'})
        #vergleich_df_list.append(temp_df_orig)

        # Geglättete Daten
        temp_df_smoothed = pd.DataFrame({epochen_spalte: df[epochen_spalte],
                                         map_spalte: smoothed_map,
                                         'Modell': f'Dota {name} (Geglättet)'})
        vergleich_df_list.append(temp_df_smoothed)
    else:
        print(f"Warnung: Benötigte Spalten ('{epochen_spalte}' oder '{map_spalte}') fehlen in {name}.")

# Die DataFrames zusammenführen
if vergleich_df_list:
    vergleich_df = pd.concat(vergleich_df_list, ignore_index=True)

    # Visualisierung mit seaborn
    plt.figure(figsize=(16, 9))
    sns.lineplot(x=epochen_spalte, y=map_spalte, hue='Modell', data=vergleich_df, marker='', linewidth=1)

    plt.xlabel('Epoch')
    plt.ylabel('mAP@50-95')
    plt.title(f'Vergleich des mAP@50-95 mit geglätteten Werten (Fenster = {window_size})')
    plt.grid(True, linestyle=':', alpha=0.7)
    sns.despine()
    plt.tight_layout()
    plt.show()
else:
    print("Keine gültigen Daten zum Plotten gefunden.")