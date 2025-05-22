import os
import ast
import csv

# Konstante Bildgröße
IMAGE_WIDTH = 2560
IMAGE_HEIGHT = 2560

# Zielverzeichnis für Labels
#OUTPUT_LABEL_DIR = "../../../scratch/tmp/t_liet02/DOTA_exp_AP_AS/train/labels"
OUTPUT_LABEL_DIR = rf"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\master_thesis\Code\temp_yolo_labels"

# CSV-Pfad (Input)
#CSV_PATH = "../../../scratch/tmp/t_liet02/airbus_datasets/plane_segmentation.csv"
CSV_PATH = rf"Code/data/airbus-plane-detection/annotations.csv"

# Klasse fix: immer "Airplane" = class_id 0
CLASS_ID = 0

def convert_to_yolo_obb(corners_pixel, img_width, img_height):
    x1_px, y1_px, x2_px, y2_px, x3_px, y3_px, x4_px, y4_px = corners_pixel

    x1_norm = x1_px / img_width
    y1_norm = y1_px / img_height
    x2_norm = x2_px / img_width
    y2_norm = y2_px / img_height
    x3_norm = x3_px / img_width
    y3_norm = y3_px / img_height
    x4_norm = x4_px / img_width
    y4_norm = y4_px / img_height

    norm_values = check_normalvalues([
        x1_norm, y1_norm, x2_norm, y2_norm,
        x3_norm, y3_norm, x4_norm, y4_norm
    ])

    return norm_values

def check_normalvalues(normalized_values):
    for i in range(len(normalized_values)):
        if normalized_values[i] < 0:
            normalized_values[i] = 0
        elif normalized_values[i] > 1:
            normalized_values[i] = 1
    return normalized_values

def transform_line(corner_list, width, height):
    # Flache Liste mit 8 Werten aus 4 Koordinatenpaaren
    flat_coords = [coord for pair in corner_list for coord in pair]
    obb_vals = convert_to_yolo_obb(flat_coords, width, height)
    string = f"{CLASS_ID} " + " ".join([f"{v:.6f}" for v in obb_vals])
    return string

def process_csv(csv_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        grouped_data = {}

        for row in reader:
            image_id = row['image_id']
            geometry_str = row['geometry']
            geometry = ast.literal_eval(geometry_str)  # z. B. [(x1,y1), (x2,y2), …]
            if image_id not in grouped_data:
                grouped_data[image_id] = []
            grouped_data[image_id].append(geometry)

        total = len(grouped_data)
        for idx, (image_id, polygons) in enumerate(grouped_data.items()):
            label_path = os.path.join(output_dir, image_id.replace(".jpg", ".txt"))
            with open(label_path, 'w') as label_file:
                for poly in polygons:
                    if len(poly) >= 4:
                        poly = poly[:4]
                        label_line = transform_line(poly, IMAGE_WIDTH, IMAGE_HEIGHT)
                        label_file.write(label_line + '\n')
            print(f"{idx}/{total} processed")

# Main aufrufen
if __name__ == "__main__":
    process_csv(CSV_PATH, OUTPUT_LABEL_DIR)
