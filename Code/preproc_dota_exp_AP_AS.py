import pandas as pd
import numpy as np
import os

def main():
    """
    Main entry point for processing the Airbus ship detection CSV file and generating YOLO OBB label files.

    - Calls process_csv_to_yolo_obb with the path to the input CSV file.
    """
    process_csv_to_yolo_obb(rf"Code\data\airbus-ship-detection\train_ship_segmentations_v2.csv")  # Ersetze durch tatsächlichen Dateipfad

# ====================
# Deine Funktionen (bereits vorhanden)
# ====================
def transform_line(arr, width, height):
    """
    Transforms a list of corner coordinates and class information into a YOLO OBB label string.

    Args:
        arr (list): List containing 8 corner coordinates, class name, and class ID.
        width (int): Image width in pixels.
        height (int): Image height in pixels.

    Returns:
        str or None: YOLO OBB label string if valid, otherwise None.
    """
    data_list = arr
    if len(data_list) == 10:
        x1, y1, x2, y2, x3, y3, x4, y4, class_name, class_id = data_list
        class_id = get_number_from_string(class_name)
        x1 = float(data_list[0])
        y1 = float(data_list[1])
        x2 = float(data_list[2])
        y2 = float(data_list[3])
        x3 = float(data_list[4])
        y3 = float(data_list[5])
        x4 = float(data_list[6]) 
        y4 = float(data_list[7])
        pixel_corner = [x1, y1, x2, y2, x3, y3, x4, y4]
        obb_vals = convert_to_yolo_obb(pixel_corner, width, height)

        string = (str(class_id)    + " " +
                  str(obb_vals[0]) + " " +
                  str(obb_vals[1]) + " " + 
                  str(obb_vals[2]) + " " + 
                  str(obb_vals[3]) + " " + 
                  str(obb_vals[4]) + " " + 
                  str(obb_vals[5]) + " " + 
                  str(obb_vals[6]) + " " + 
                  str(obb_vals[7]))
        return string
    else:
        print(f"Warning: Line skipped due to invalid data: {arr}")
        return None

def get_number_from_string(input_string):
    """
    Maps a class name string to its corresponding numeric class ID.

    Args:
        input_string (str): Class name.

    Returns:
        int or None: Numeric class ID, or None if not found.
    """
    class_mapping = {
        "plane": 0, "ship": 1, "storage-tank": 2, "baseball-diamond": 3,
        "tennis-court": 4, "basketball-court": 5, "ground-track-field": 6,
        "harbor": 7, "bridge": 8, "large-vehicle": 9, "small-vehicle": 10,
        "helicopter": 11, "roundabout": 12, "soccer-ball-field": 13,
        "swimming-pool": 14, "container-crane": 15
    }
    return class_mapping.get(input_string.lower())

def convert_to_yolo_obb(corners_pixel, img_width, img_height):
    """
    Converts pixel corner coordinates to normalized YOLO OBB format.

    Args:
        corners_pixel (list): List of 8 pixel coordinates (x1, y1, ..., x4, y4).
        img_width (int): Image width in pixels.
        img_height (int): Image height in pixels.

    Returns:
        list: List of 8 normalized corner coordinates.
    """
    x1_px, y1_px, x2_px, y2_px, x3_px, y3_px, x4_px, y4_px = corners_pixel
    norm_vals = [
        x1_px / img_width, y1_px / img_height,
        x2_px / img_width, y2_px / img_height,
        x3_px / img_width, y3_px / img_height,
        x4_px / img_width, y4_px / img_height
    ]
    return check_normalvalues(norm_vals, corners_pixel)

def check_normalvalues(normalized_values, cp):
    """
    Ensures all normalized values are within [0, 1] range.

    Args:
        normalized_values (list): List of normalized coordinates.
        cp (list): Original pixel coordinates (unused).

    Returns:
        list: Corrected list of normalized coordinates.
    """
    for i in range(len(normalized_values)):
        if normalized_values[i] < 0:
            normalized_values[i] = 0
        elif normalized_values[i] > 1:
            normalized_values[i] = 1
    return normalized_values

# ====================
# RLE Decoding und Label-Erzeugung
# ====================
def rle_decode(mask_rle, shape):
    """
    Decodes a run-length encoded (RLE) mask string into a binary mask array.

    Args:
        mask_rle (str): RLE string encoding the mask.
        shape (tuple): Shape of the output mask (height, width).

    Returns:
        np.ndarray: Decoded binary mask of the specified shape.
    """
    s = list(map(int, mask_rle.split()))
    starts, lengths = s[::2], s[1::2]
    starts = np.array(starts) - 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((shape[1], shape[0])).T

def mask_to_4corners(mask):
    """
    Extracts the four corner coordinates of the bounding box from a binary mask.

    Args:
        mask (np.ndarray): Binary mask array.

    Returns:
        list or None: List of 8 coordinates (x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max) or None if mask is empty.
    """
    y, x = np.where(mask)
    if len(x) == 0 or len(y) == 0:
        return None
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    return [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]  # clockwise

# ====================
# Main: Verarbeitung der CSV
# ====================
def process_csv_to_yolo_obb(csv_path, output_dir="yolo_labels", img_width=768, img_height=768):
    """
    Processes the input CSV file, decodes masks, extracts bounding box corners, and writes YOLO OBB label files.

    Args:
        csv_path (str): Path to the input CSV file.
        output_dir (str, optional): Directory to save YOLO label files. Default is "yolo_labels".
        img_width (int, optional): Image width in pixels. Default is 768.
        img_height (int, optional): Image height in pixels. Default is 768.

    Side effects:
        Writes YOLO OBB label files for each image in the output directory.
        Prints progress information to the console.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["EncodedPixels"])
    counter = 0

    for image_id, group in df.groupby("ImageId"):
        label_path = os.path.join(output_dir, image_id.replace(".jpg", ".txt"))
        with open(label_path, "w") as f:
            for _, row in group.iterrows():
                
                rle = row["EncodedPixels"]
                mask = rle_decode(rle, (img_height, img_width))
                corners = mask_to_4corners(mask)
                if corners:
                    corners_with_class = corners + ["ship", 1]
                    label_line = transform_line(corners_with_class, img_width, img_height)
                    if label_line:
                        f.write(label_line + "\n")
            counter = counter +1 
            print(str(counter)+"/" + str(len(df.groupby("ImageId"))))

# ====================
# Aufruf
# ====================
main()

