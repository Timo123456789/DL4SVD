import cv2
import numpy as np
from scipy.spatial import ConvexHull
import os
import sys

def main():
    """
    Main entry point for transforming DOTA label files to YOLO OBB format.

    - Sets the path to the DOTA dataset.
    - Calls transform_label_folder for the training set.
    - Prints a success message when transformation is complete.
    """
    path_dota = rf'Code\data\DOTA'

    transform_label_folder(path_dota,"train")

    print("transform successfull")


def transform_label_folder(folder_path, string_tag):
    """
    Transforms all label files in a given folder to YOLO OBB format.

    Args:
        folder_path (str): Path to the DOTA dataset folder.
        string_tag (str): Subset tag ("train" or "test").

    Side effects:
        Creates necessary output folders.
        Processes each label file and writes transformed labels.
        Prints progress and error messages.
    """
    folder_path = rf"{folder_path}\{string_tag}\labels" 
    counter = 0

    path_obj = create_folder_structure()
    if string_tag == "test":

        print("test")

        json_path= rf"Code\data\DOTA\test\test_info.json"





        sys.exit()

    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                transform_label(file_path, string_tag, path_obj)
            print(str(counter)+"/" + str(len(os.listdir(folder_path))))
            counter += 1
            


                
    except FileNotFoundError:
        print(f"Error: The folder '{folder_path}' was not found.")
    except NotADirectoryError:
        print(f"Error: '{folder_path}' is not a valid folder.")

def transform_label(filepath, string_tag, path_obj):
    """
    Transforms a single label file to YOLO OBB format and writes the result.

    Args:
        filepath (str): Path to the label file.
        string_tag (str): Subset tag ("train" or "test").
        path_obj (dict): Dictionary with output folder paths.

    Side effects:
        Reads the corresponding image to get dimensions.
        Writes the transformed label to the output folder.
        Prints warnings for malformed lines or write errors.
    """
    new_labels=[]
   
    new_labels_path = filepath.replace(rf'{string_tag}\label', rf'new_labels\{string_tag}\label')
    image_path = filepath.replace('label','image')
    image_path = image_path.replace('.txt', '.png')

    height, width, channels = cv2.imread(image_path).shape
    
  
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i in lines:
            values = i.split()
            if len(values) == 10:
                transformed_line = transform_line(values, width, height)
                if transformed_line != None:
                    new_labels.append(transformed_line)

    
                
            
    
    
    try:
        with open(new_labels_path, 'w', encoding='utf-8') as file:
            for entry in new_labels:
                file.write(entry + '\n')
            #print(f"The array was successfully written to the file '{filepath}'.")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")
    return 0



def create_folder_structure():
    """
    Creates the output folder structure for transformed labels and images.

    Returns:
        dict: Dictionary with paths for train/val images and labels.

    Side effects:
        Creates directories if they do not exist.
        Prints status messages about directory creation.
    """

    def make_directories(path):
        try:
            os.makedirs(path)
            print(f"Ordnerstruktur '{path}' erfolgreich erstellt.")
        except FileExistsError:
            print(f"Einige oder alle Ordner in '{path}' existieren bereits.")
        except Exception as e:
            print(f"Ein Fehler ist aufgetreten: {e}")

    
    path_obj = {
        'path_train_images' : f"code/data/DOTA/new_labels/train/images",
        'path_train_labels' : f"code/data/DOTA/new_labels/train/labels",
        'path_val_images' : f"code/data/DOTA/new_labels/val/images",
        'path_val_labels' : f"code/data/DOTA/new_labels/val/labels",
    }
  
    
    #make_directories(path_obj['path_train_images'])
    make_directories(path_obj['path_train_labels'])
    
    #make_directories(path_obj['path_val_images'])
    make_directories(path_obj['path_val_labels'])
   
    return path_obj
    
def transform_line(arr, width, height):
    """
    Transforms a list of DOTA label values into a YOLO OBB label string.

    Args:
        arr (list): List of 10 elements (8 coordinates, class name, class id).
        width (int): Image width in pixels.
        height (int): Image height in pixels.

    Returns:
        str or None: YOLO OBB label string if valid, otherwise None.

    Prints:
        Warning if the line does not have the expected number of elements.
    """
    
    data_list=arr
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
        return f'{class_id}\'{x1}\'\'{y1}\'\'{x2}\'\'{y2}\'\'{x3}\'\'{y3}\'\'{x4}\'\'{y4}'
    else:
        print(f"Warning: Line does not have the expected number of elements and will be skipped: {arr}")
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
    return class_mapping.get(input_string.lower())


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

    x1_norm = x1_px / img_width
    y1_norm = y1_px / img_height
    x2_norm = x2_px / img_width
    y2_norm = y2_px / img_height
    x3_norm = x3_px / img_width
    y3_norm = y3_px / img_height
    x4_norm = x4_px / img_width
    y4_norm = y4_px / img_height

    norm_values = check_normalvalues([x1_norm, y1_norm, x2_norm, y2_norm, x3_norm, y3_norm, x4_norm, y4_norm], corners_pixel)

    return norm_values


def check_normalvalues(normalized_values, cp):
    """
    Checks a list of normalized values and clamps them to [0, 1] if out of bounds.

    Args:
        normalized_values (list): List of normalized coordinates.
        cp (list): Original pixel coordinates (unused).

    Returns:
        list: Corrected list of normalized coordinates.
    """
    found = False
    cp_normalized_values = normalized_values
    for i in range(len(cp_normalized_values)): # Iteriere ueber die Indizes der Liste
        if cp_normalized_values[i] < 0:
            cp_normalized_values[i] = 0
            found = True
        elif cp_normalized_values[i] > 1:
            cp_normalized_values[i] = 1
            #found = True

    # if found:
    #     print(cp_normalized_values)
    return cp_normalized_values
    #raise ValueError(f"Ungueltiger Normwert gefunden: {wert}. Normwerte muessen zwischen 0 und 1 liegen.")

main()