import os
import pandas as pd

# =====================
# Transformation Logik
# =====================

def transform_line(arr, width, height):
    """
    Transforms a list of corner coordinates and class information into a YOLO OBB label string.

    Args:
        arr (list): List containing 8 corner coordinates, class name, and a placeholder value.
        width (int): Image width in pixels.
        height (int): Image height in pixels.

    Returns:
        str or None: YOLO OBB label string if valid, otherwise None.

    Prints:
        Warning if the input list does not have the expected length.
    """
    if len(arr) == 10:
        x1, y1, x2, y2, x3, y3, x4, y4, class_name, _ = arr
        class_id = get_number_from_string(class_name)
        pixel_corner = [float(x1), float(y1), float(x2), float(y2),
                        float(x3), float(y3), float(x4), float(y4)]
        obb_vals = convert_to_yolo_obb(pixel_corner, width, height)

        string = (f"{class_id} " +
                  " ".join([f"{v:.6f}" for v in obb_vals]))
        return string
    else:
        print(f"Warning: Zeile mit unerwarteter Länge wird übersprungen: {arr}")
        return None

def get_number_from_string(input_string):
    """
    Maps a class name string to its corresponding numeric class ID.

    Args:
        input_string (str): Class name.

    Returns:
        int: Numeric class ID, or -1 if not found.
    """
    class_mapping = {
        "plane": 0,
        "ship": 1,
        "storage-tank": 2,
        "baseball-diamond": 3,
        "tennis-court": 4,
        "basketball-court": 5,
        "ground-track-field": 6,
        "harbor": 7,
        "bridge": 8,
        "large-vehicle": 9,
        "small-vehicle": 10,
        "helicopter": 11,
        "roundabout": 12,
        "soccer-ball-field": 13,
        "swimming-pool": 14,
        "container-crane": 15
    }
    return class_mapping.get(input_string.lower(), -1)

def convert_to_yolo_obb(corners_pixel, img_width, img_height):
    """
    Converts pixel corner coordinates to normalized YOLO OBB format.

    Args:
        corners_pixel (list): List of 8 pixel coordinates (x1, y1, ..., x4, y4).
        img_width (int): Image width in pixels.
        img_height (int): Image height in pixels.

    Returns:
        list: List of 8 normalized corner coordinates in [0, 1].
    """
    x1_px, y1_px, x2_px, y2_px, x3_px, y3_px, x4_px, y4_px = corners_pixel
    norm_values = [x1_px / img_width, y1_px / img_height,
                   x2_px / img_width, y2_px / img_height,
                   x3_px / img_width, y3_px / img_height,
                   x4_px / img_width, y4_px / img_height]
    return check_normalvalues(norm_values)

def check_normalvalues(normalized_values):
    """
    Checks a list of normalized values and clamps them to [0, 1] if out of bounds.

    Args:
        normalized_values (list): List of normalized coordinates.

    Returns:
        list: Corrected list of normalized coordinates.
    """
    corrected = []
    for val in normalized_values:
        if val < 0:
            corrected.append(0.0)
        elif val > 1:
            corrected.append(1.0)
        else:
            corrected.append(val)
    return corrected

# =====================
# CSV-Verarbeitung
# =====================

def transform_label_csv(csv_path, output_label_folder):
    """
    Reads a CSV file with bounding box coordinates and class names, transforms each row to YOLO OBB format,
    and writes the results to individual label files.

    Args:
        csv_path (str): Path to the input CSV file.
        output_label_folder (str): Directory to save YOLO label files.

    Side effects:
        Creates the output directory if it does not exist.
        Writes YOLO OBB label files for each image.
        Prints progress and error messages.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Fehler beim Lesen der CSV-Datei: {e}")
        return

    if not os.path.exists(output_label_folder):
        os.makedirs(output_label_folder)

    total = len(df)

    for idx, row in df.iterrows():
        try:
            image_id = row['ImageId']
            label_filename = os.path.splitext(image_id)[0] + ".txt"
            label_path = os.path.join(output_label_folder, label_filename)

            x1, y1 = row['x1'], row['y1']
            x2, y2 = row['x2'], row['y2']
            x3, y3 = row['x3'], row['y3']
            x4, y4 = row['y4'], row['y4']
            class_name = row['class_name']

            width, height = 768, 768  # fix laut Vorgabe
            data_list = [x1, y1, x2, y2, x3, y3, x4, y4, class_name, 0]

            transformed_line = transform_line(data_list, width, height)
            if transformed_line:
                with open(label_path, 'a') as f:
                    f.write(transformed_line + '\n')

            print(f"{idx}/{total} processed")

        except Exception as e:
            print(f"Fehler bei Zeile {idx}: {e}")


# =====================
# Einstiegspunkt
# =====================

def main():
    """
    Main entry point for processing the Airbus ship segmentation CSV file and generating YOLO OBB label files.

    - Sets the input CSV path and output label directory.
    - Calls transform_label_csv to process and transform the data.
    - Prints a completion message.
    """
    csv_path = "../../../scratch/tmp/t_liet02/airbus_datasets/ship_segmentation.csv"
    output_label_dir = "../../../scratch/tmp/t_liet02/DOTA_exp_AP_AS/train/labels"

    transform_label_csv(csv_path, output_label_dir)
    print("Transformation abgeschlossen.")

if __name__ == "__main__":
    main()
