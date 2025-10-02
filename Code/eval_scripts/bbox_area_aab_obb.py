import cv2
import numpy as np
from scipy.spatial import ConvexHull
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

REMOVE_TOP_N = 754

def main():
    """
    Main function to compare and visualize the areas of Oriented Bounding Boxes (OBB) and Axis-Aligned Bounding Boxes (AAB).
    This function performs the following steps:
    1. Loads bounding box data from specified directories for OBB, AAB, and old AAB formats.
    2. Prints sample data and counts for each bounding box type.
    3. Calculates the area for each bounding box type using appropriate functions.
    4. Removes the largest N areas from each list to mitigate outlier effects.
    5. Prints sample area values for each bounding box type.
    6. Plots boxplots to visualize the area distributions of OBB, AAB, and old AAB bounding boxes.
    Assumes the existence of the following functions and variables:
    - get_bounding_boxes_as_array(path, is_aab)
    - calculate_oriented_bounding_box_area(box)
    - calculate_aab_area(box)
    - plot_area_boxplots(obb_areas, aab_areas, aab_old_areas)
    - REMOVE_TOP_N (int): Number of largest areas to remove from each list.
    """
    print("test")
    path_obb = r'Code\data\folds\data\cross_validation\obb'
    path_aab= r'Code\data\folds\data\cross_validation\aab'
    path_aab_old = r'Code\data\folds\data\cross_validation\aab_old'

    obb_boxes = get_bounding_boxes_as_array(path_obb, False)
    aab_boxes = get_bounding_boxes_as_array(path_aab, True)
    aab_old_boxes = get_bounding_boxes_as_array(path_aab_old, True)

 
    print("aab:")
    print(aab_boxes[:2])
    print(f"Anzahl aab_boxes: {len(aab_boxes)}")
    print(f"Anzahl obb_boxes: {len(obb_boxes)}")
    print(f"Anzahl aab_old_boxes: {len(aab_old_boxes)}")


    obb_areas = [calculate_oriented_bounding_box_area(box) for box in obb_boxes]

 

    # Flächen für AABs berechnen (AAB und aab_old)
    aab_areas = [calculate_aab_area(box) for box in aab_boxes]
    aab_old_areas = [calculate_aab_area(box) for box in aab_old_boxes]

    # Die größten N Elemente aus jedem Array entfernen
    obb_areas = sorted(obb_areas)[:-REMOVE_TOP_N] if len(obb_areas) > REMOVE_TOP_N else obb_areas
    aab_areas = sorted(aab_areas)[:-REMOVE_TOP_N] if len(aab_areas) > REMOVE_TOP_N else aab_areas
    aab_old_areas = sorted(aab_old_areas)[:-REMOVE_TOP_N] if len(aab_old_areas) > REMOVE_TOP_N else aab_old_areas

    print("obb:")
    print(obb_areas[:2])
    print("aab:")
    print(aab_areas[:2])
    print("aab_old")
    print(aab_old_areas[:2])

    plot_area_boxplots(obb_areas, aab_areas, aab_old_areas)


def calculate_aab_area(box):
    """
    Calculates the area of an axis-aligned bounding box (AAB).

    Parameters:
        box (list or tuple): The bounding box coordinates. Can be either:
            - [x1, y1, x2, y2]: where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
            - [[x1, y1], [x2, y2]] or [(x1, y1), (x2, y2)]: where each element is a coordinate pair.

    Returns:
        float: The area of the bounding box.
    """
    # Erwartet [x1, y1, x2, y2]
    if isinstance(box[0], list) or isinstance(box[0], tuple):
        x1, y1 = box[0]
        x2, y2 = box[1]
    else:
        x1, y1, x2, y2 = box
    return abs(x2 - x1) * abs(y2 - y1)

def get_bounding_boxes_as_array(path, aab_old_bool):
    """
    Reads bounding box data from train, validation, and test label folders within a specified path,
    and returns all bounding box data as a combined list or array.
    Args:
        path (str): The root directory containing the 'fold0' subdirectory with 'train', 'val', and 'test' label folders.
        aab_old_bool (bool): A flag indicating which format or version of bounding box data to read.
    Returns:
        list: A list containing bounding box data from all train, validation, and test label files.
    """
    path_train = os.path.join(path, 'fold0','train', 'labels')
    path_val = os.path.join(path, 'fold0', 'val', 'labels')
    path_test = os.path.join(path, 'fold0','test', 'labels')

    data_train = read_all_text_files_from_folder(path_train, aab_old_bool)
    data_val = read_all_text_files_from_folder(path_val, aab_old_bool)
    data_test = read_all_text_files_from_folder(path_test, aab_old_bool)
    all_data = data_train + data_val + data_test

    return all_data

 

def read_all_text_files_from_folder(folder_path, aab_old_bool):
    """
    Reads all text files from the specified folder and extracts bounding box coordinates.
    The function supports two bounding box formats:
    - Oriented Bounding Box (OBB): 8 normalized coordinates (x1, y1, ..., x4, y4)
    - Axis-Aligned Bounding Box (AAB): 4 normalized coordinates (cx, cy, w, h)
    For each file ending with ".txt", the function parses each line, removes the first element (typically the class label),
    and converts the normalized coordinates to pixel values using a fixed image size (1024x1024).
    Args:
        folder_path (str): Path to the folder containing text files.
        aab_old_bool (bool): If True, handles YOLO format (class cx cy w h) for AAB boxes.
    Returns:
        list: A list of bounding boxes in pixel coordinates. Each bounding box is represented as:
            - OBB: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            - AAB: [x1, y1, x2, y2]
    """
    all_boxes = []
    img_width, img_height = 1024, 1024
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        parts = parts[1:]  # Erstes Element entfernen
                    # OBB: x1,y1,x2,y2,x3,y3,x4,y4 (normalisiert)
                    if len(parts) == 8:
                        # coords = list(map(float, parts))
                        # pixel_coords = []
                        # for i in range(0, 8, 2):
                        #     x = coords[i] * img_width
                        #     y = coords[i+1] * img_height
                        #     pixel_coords.append([x, y])
                        # all_boxes.append(pixel_coords)

                        coords = list(map(float, parts))
                        x1 = coords[0] * img_width
                        y1 = coords[1] * img_height
                        x2 = coords[2] * img_width
                        y2 = coords[3] * img_height
                        x3 = coords[4] * img_width
                        y3 = coords[5] * img_height
                        x4 = coords[6] * img_width
                        y4 = coords[7] * img_height
                        all_boxes.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                    # AAB: x1,y1,x2,y2 (normalisiert)
                    elif len(parts) == 4:
                        # coords = list(map(float, parts))
                        # x1 = coords[0] * img_width
                        # y1 = coords[1] * img_height
                        # x2 = coords[2] * img_width
                        # y2 = coords[3] * img_height
                        # all_boxes.append([x1, y1, x2, y2])

                        cx, cy, w, h = map(float, parts)
                        x_center = cx * img_width
                        y_center = cy * img_height
                        box_width = w * img_width
                        box_height = h * img_height
                        x1 = x_center - box_width / 2
                        y1 = y_center - box_height / 2
                        x2 = x_center + box_width / 2
                        y2 = y_center + box_height / 2
                        all_boxes.append([x1, y1, x2, y2])
                    # Optional: YOLO Format (class cx cy w h)
                    elif len(parts) == 4 and aab_old_bool == True:
                        cx, cy, w, h = map(float, parts[1:5])
                        x_center = cx * img_width
                        y_center = cy * img_height
                        box_width = w * img_width
                        box_height = h * img_height
                        x1 = x_center - box_width / 2
                        y1 = y_center - box_height / 2
                        x2 = x_center + box_width / 2
                        y2 = y_center + box_height / 2
                        all_boxes.append([x1, y1, x2, y2])
    return all_boxes


def calculate_bounding_box_area(points):
    """
    Calculates the area of the axis-aligned bounding box (AABB) that encloses a set of 2D points.
    Assumes points are provided as a list or array of [y, x] or [x, y] coordinates.
    If the input format is [y, x], the function will automatically transpose to [x, y] if needed.
    Parameters:
        points (list or np.ndarray): A list or numpy array of shape (n, 2), where each row represents a point.
    Returns:
        int: The area (number of pixels) of the bounding box. Returns 0 if points is None or empty.
    Raises:
        ValueError: If the input does not have shape (n, 2).
    """
   
    if points is None or len(points) < 1:
        return 0
    print(points)
    points_np = np.array(points)

    # Annahme: Punkte sind im Format [y, x]. Wenn nicht, transponieren wir.
    # ÃœberprÃ¼fen anhand des Mittelwerts, welche Spalte tendenziell grÃ¶ÃŸere Werte hat (oft x).
    if points_np.shape[1] == 2 and np.mean(points_np[:, 1]) > np.mean(points_np[:, 0]):
        points_np = points_np[:, [1, 0]]  # Spalten tauschen: [x, y] -> [y, x]

    if points_np.shape[1] != 2:
        raise ValueError("Die Eingabepunkte mÃ¼ssen die Form (n, 2) haben.")

    min_y = int(np.min(points_np[:, 0]))
    max_y = int(np.max(points_np[:, 0]))
    min_x = int(np.min(points_np[:, 1]))
    max_x = int(np.max(points_np[:, 1]))

    
    breite = max_x - min_x + 1
    hoehe = max_y - min_y + 1

    print(breite)
    print(hoehe)

    anzahl_pixel = breite * hoehe
    return anzahl_pixel


def calculate_oriented_bounding_box_area(points):
    """
    Calculates the area of the minimum oriented bounding box (OBB) that encloses a set of 2D points.
    The function first computes the convex hull of the input points, then finds the smallest rectangle
    (possibly rotated) that can contain all the hull points using OpenCV's minAreaRect.
    Args:
        points (array-like): An iterable of 2D points with shape (n, 2), where n >= 3.
    Returns:
        float: The area of the oriented bounding box. Returns 0.0 if fewer than 3 points are provided.
    Raises:
        ValueError: If the input points do not have shape (n, 2).
    """
    
    points_np = np.array(points)
    if points_np.shape[0] < 3:
        return 0.0

    if points_np.shape[1] != 2:
        raise ValueError("Die Eingabepunkte mÃ¼ssen die Form (n, 2) haben.")

    # Finde die konvexe HÃ¼lle der Punkte
    hull = ConvexHull(points_np)
    hull_points = points_np[hull.vertices]

    # Finde die orientierte Bounding Box mit OpenCV
    rect = cv2.minAreaRect(hull_points.astype(np.float32))
    width, height = rect[1]
    area = width * height
    return area

def plot_area_boxplots(obb_areas, aab_areas, aab_old_areas):
    """
    Plots and saves boxplots comparing the areas of different bounding box types.
    Parameters:
        obb_areas (list or array-like): Areas of Oriented Bounding Boxes (OBB).
        aab_areas (list or array-like): Areas of Axis-Aligned Bounding Boxes (ABB).
        aab_old_areas (list or array-like): Areas of ABBs computed within OBBs.
    The function creates a boxplot using Seaborn to visualize the distribution of areas for each bounding box type.
    The plot is saved as an SVG file and displayed. Axis labels and tick sizes are customized for presentation.
    """
       # DataFrame for Seaborn
    df = pd.DataFrame({
        "Area": obb_areas + aab_areas + aab_old_areas,
        "Type": (["OBB"] * len(obb_areas)) 
              + (["ABB"] * len(aab_areas)) 
              + (["ABB in OBB"] * len(aab_old_areas))
    })

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=df, x="Type", y="Area",
        palette={"OBB": "orange", "ABB": "steelblue", "ABB in OBB": "green"}
    )

    # Schriftgrößen anpassen
    ax.set_ylabel("Area (pixels)", fontsize=14)
    ax.set_xlabel("Bounding Box Type", fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    #ax.set_title("Boxplot of Areas per Bounding Box Type", fontsize=16)

    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\master_thesis\MA-Thesis-Latex\images\015Results\01abb_vs_obb\boxplot_areas.svg", format="svg", transparent=True)
    plt.show()

if __name__ == "__main__":
    main()
    # Nach main() die Arrays holen (ggf. als Rückgabewert aus main() machen)
    # Beispiel: plot_area_boxplots(obb_areas, aab_areas, aab_old_areas)