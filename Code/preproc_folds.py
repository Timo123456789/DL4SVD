import cv2
import numpy as np
from scipy.spatial import ConvexHull
import os

def main():
    oriented = False
    ir = False
    bool_create_yaml = False
    limiter = None
    #path_folds = r'Code\data\folds\txts'
    path_all_images = r'Code\data\all_vedai_images'
    path_labels = r'Code\data\all_vedai_images\annotation.txt'
   
    #lines_fold = read_file(path_folds)
    test_rgb_img = r'Code\data\all_vedai_images\00000124_co.png'
    test_ir_img = r'Code\data\all_vedai_images\00000124_ir.png'
    destination_path = r'Code\data\all_vedai_images\00000124_ir.png'

    #merge_RGB_and_IR_image(test_rgb_img, test_ir_img, destination_path)

    
    #show_every_picture_with_oriented_bounding_box(path_all_images,r'Code\data\folds\txts\fold01.txt',path_labels,oriented, False,False)

    #create_all_folds(path_all_images,path_folds, path_labels, ir, oriented, bool_create_yaml)
    
    #show_one_picture_with_yolo_label(1,'00000010', False, "val")
    create_fold(1,path_all_images,path_labels, ir, False, bool_create_yaml, limiter, oriented)

    # print("RGB Folds successful created")

def show_one_picture_with_yolo_label(fold_nr,img_nr,ir, string_tag):
    if ir == True:
        image_path = rf'Code\data\folds\data\fold{fold_nr}\{string_tag}\images\{img_nr}_co.png'
    else:
        image_path = rf'Code\data\folds\data\fold{fold_nr}\{string_tag}\images\{img_nr}_co.png'
    label_path = rf'Code\data\folds\data\fold{fold_nr}\{string_tag}\labels\{img_nr}_co.txt'

  
    print(label_path)
    img = cv2.imread(image_path)

    labels = read_file(label_path)
    for label in labels:
        img = add_labeled_normalized_coordinates_as_points(image_path, label)

    cv2.imshow("Bild mit Punkten", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def add_labeled_normalized_coordinates_as_points(image_path, labels_string, point_color=(0, 0, 255), point_radius=5, point_thickness=-1):
    """
    Fügt normierte Koordinaten aus einer Label-Zeichenkette als farbige Punkte in ein Bild ein.
    Das erste Element der Label-Zeile wird übersprungen.

    Args:
        image_path (str): Der Pfad zum Bild.
        labels_string (str): Eine Zeichenkette mit den Labels und normierten Koordinaten,
                             z.B. "8 0.32421875 0.1240234375 ...".
                             Es wird erwartet, dass die Koordinaten paarweise folgen.
        point_color (tuple): Die Farbe der Punkte im BGR-Format (Standard: Rot).
        point_radius (int): Der Radius der Punkte in Pixeln (Standard: 5).
        point_thickness (int): Die Dicke der Punkte (-1 für gefüllte Kreise, Standard: -1).
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Fehler: Bild nicht gefunden unter {image_path}")
            return

        height, width, _ = img.shape

        # Teile die Label-Zeichenkette auf und überspringe das erste Element
        parts = labels_string.split()
        coordinates_str = parts[1:]

        # Konvertiere die Koordinaten-Strings in Floats
        coordinates = [float(coord) for coord in coordinates_str]

        # Konvertiere die flache Liste der Koordinaten in Paare
        points = []
        for i in range(0, len(coordinates), 2):
            if i + 1 < len(coordinates):
                x_norm = coordinates[i]
                y_norm = coordinates[i+1]
                x_pixel = int(x_norm * width)
                y_pixel = int(y_norm * height)
                points.append((x_pixel, y_pixel))
            else:
                print("Warnung: Ungerade Anzahl von Koordinaten. Letzte Koordinate wird ignoriert.")
                break
      

        for p in points:
            cv2.circle(img, p, point_radius, point_color, point_thickness)

        return img

    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

def create_all_folds(path_all_images, path_labels, ir, oriented,bool_create_yaml, limiter):
    fold_nr = 1


    for fold_nr in range(10):
        if fold_nr != 0:
            create_fold(fold_nr,path_all_images,path_labels, ir, False, bool_create_yaml, limiter)

        #except:
        #    print("No fold number found")
    create_fold(10,path_all_images,path_labels,False, True, bool_create_yaml)


def merge_RGB_and_IR_image(rgb_path, ir_path, destination_path):
    # Load the RGB image
    rgb_img = cv2.imread(rgb_path)

    # Load the IR image (as grayscale, as IR is typically a single intensity band)
    ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

    if rgb_img is None or ir_img is None:
         print("One or both images could not be loaded.")
    elif rgb_img.shape[:2] != ir_img.shape[:2]:
         print("RGB and IR images must have the same height and width.")
    else:
    # Split the RGB image into its individual channels (Blue, Green, Red)
        b, g, r = cv2.split(rgb_img)

        # Add the IR image as the fourth band
        merged_img = cv2.merge([b, r, g, ir_img])

        # The resulting image now has 4 channels: Blue, Green, Red, Infrared.
        # Note that the interpretation and visualization of such a
        # 4-channel image depend on the subsequent processing.
        # Standard image viewers usually expect RGB or RGBA.

        # Save the image with the added IR band (e.g., as PNG, which supports alpha)
        # Even though we call it 'merged_img', it's essentially a 4-channel image.
        cv2.imwrite('rgb_ir_merged.png', merged_img)
        #print("RGB image with IR band successfully saved.")

        return merged_img

        # Displaying the image is not trivial as it has 4 channels that cannot be
        # directly interpreted as RGB. You would need to decide how to combine
        # these 4 channels for visualization (e.g., mapping IR to a color channel).
        # Here, we only show it as a raw 4-channel array (which might not look meaningful).
        # cv2.imshow('RGB + IR', merged_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

def create_fold(fold_nr,path_all_images,path_labels, ir, fold10bool, bool_create_yaml, limiter, oriented):
    def create_image_and_label(lines,  string_tag):
        counter = 0
        for line in lines:
            counter += 1
            target = line
            if ir == True:
                image_path = f"{path_all_images}/{target}_ir.png"
            else:
                image_path = f"{path_all_images}/{target}_co.png"

            filtered_labels = select_all_labels_in_img(target, labels)
            copy_image(image_path, paths_object['path_'+string_tag+'_images'])
            create_label_file(target, filtered_labels, paths_object['path_'+string_tag+'_labels'], image_path, ir, oriented, string_tag)
            print("Fold Nr:"+str(fold_nr)+" /  "+ string_tag+" image: "+str(counter) + "/" + str(len(lines)))
            
            if limiter != None:
                if counter == limiter and string_tag == 'train':
                    print("Limiter! BREAK "+ string_tag+" at " + str(limiter))
                    break
                elif counter == (limiter/10) and string_tag == 'val':
                    print("Limiter! BREAK "+ string_tag+" at " + str(limiter))
                break
            

    if fold10bool == True:
        fold_train_images_path = rf'Code\data\folds\txts\fold10.txt'
        fold_val_images_path = rf'Code\data\folds\txts\fold10test.txt'
        yaml_path = rf'Code\data\folds\data\fold10'
    else:
        fold_train_images_path = rf'Code\data\folds\txts\fold0{fold_nr}.txt'
        fold_val_images_path = rf'Code\data\folds\txts\fold0{fold_nr}test.txt'
        yaml_path = rf'Code\data\folds\data\fold{fold_nr}'

    paths_object = create_folder_structure(fold_nr)

    lines_fold_train = read_file(fold_train_images_path)
    lines_fold_val = read_file(fold_val_images_path)
    labels = read_file(path_labels)
    
    if bool_create_yaml:
        create_yaml(yaml_path)
        print("Yaml successfull created.")
    

    create_image_and_label(lines_fold_train, "train")
    create_image_and_label(lines_fold_val, "val")
   
    print("Fold " + str(fold_nr) + " successfull created.")
    return 0




   


def create_label_file(target, labels, path, img_path, ir, oriented, string_tag):
    def convert_class_to_yolo(class_id):
        if class_id == '001':
            label = 'Car'
            return str(0)
        elif class_id == '002':
            label = 'Truck'
            return str(1)
        elif class_id == '023':
            label = 'Ship'
            return str(2)
        elif class_id == '004':
            label = 'Tractor'
            return str(3)
        elif class_id == '005':
            label = 'Camping Car'
            return str(4)
        elif class_id == '009':
            label = 'van'
            return str(5)
        elif class_id == '010':
            label = 'vehicle'
            return str(6)
        elif class_id == '011':
            label = 'pick-up'
            return str(7)
        elif class_id == '031':
            label = 'plane'
            return str(8)
        else:
            return str(6)
        return 0

    if ir == True:
        file_path = f"{path}/{target}_ir.txt"
    else:
        file_path = f"{path}/{target}_co.txt"

    img = cv2.imread(img_path)

    
  
    try:
        with open(file_path, 'w') as file:
            for label in labels:
                transf_label = calc_pixel_like_authors(img, label, oriented, True) 
                px_corner = [transf_label[1][1],transf_label[1][0],transf_label[2][1],transf_label[2][0],transf_label[3][1],transf_label[3][0],transf_label[4][1],transf_label[4][0]]

                if oriented == True:
                    yolo_transf_label = convert_to_yolo_obb(px_corner, img.shape[1], img.shape[0])
                    #print(convert_class_to_yolo(transf_label[0]))
                    #if string_tag == "train":
                    file_string = (convert_class_to_yolo(transf_label[0]) + " "+
                        str(yolo_transf_label[0])+" "+
                        str(yolo_transf_label[1])+" "+
                        str(yolo_transf_label[2])+" "+ 
                        str(yolo_transf_label[3])+" "+ 
                        str(yolo_transf_label[4])+" "+ 
                        str(yolo_transf_label[5])+" "+
                        str(yolo_transf_label[6])+" "+
                        str(yolo_transf_label[7]) + '\n')
                elif oriented == False:
                    yolo_transf_label = convert_label_to_yolo_classic(px_corner,img.shape[1], img.shape[0])
                    file_string = (convert_class_to_yolo(transf_label[0]) + " "+
                        str(yolo_transf_label[0])+" "+
                        str(yolo_transf_label[1])+" "+
                        str(yolo_transf_label[2])+" "+ 
                        str(yolo_transf_label[3]) + '\n')
                    
                file.write(file_string)
            #datei.write(inhalt)
        #print(f"Die Datei '{file_path}' wurde erfolgreich mit Inhalt erstellt.")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten in der create_label_file: {e}")
    
def convert_label_to_yolo_classic(corners_pixel, img_width, img_height):
    """
    Converts the coordinates of four pixels defining a polygon
    into the YOLOv9 label format.

    Args:
    x1, y1, x2, y2, x3, y3, x4, y4: Coordinates of the four corner points of the polygon.
    image_width (int): Width of the image in pixels.
    image_height (int): Height of the image in pixels.

    Returns:
    str: A YOLOv9-compliant label string in the format
        'class_id center_x center_y width height', where the coordinates
        are normalized.
    """

    x1=corners_pixel[0]
    y1=corners_pixel[1]
    x2=corners_pixel[2]
    y2=corners_pixel[3]
    x3=corners_pixel[4]
    y3=corners_pixel[5]
    x4=corners_pixel[6]
    y4=corners_pixel[7]
    # Find the minimum and maximum x and y values to define the bounding box
    min_x = min(x1, x2, x3, x4)
    max_x = max(x1, x2, x3, x4)
    min_y = min(y1, y2, y3, y4)
    max_y = max(y1, y2, y3, y4)

    # Calculate the center of the bounding box
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    # Calculate the width and height of the bounding box
    width = max_x - min_x
    height = max_y - min_y

    # Normalize the values to the range [0, 1]
    center_x_normalized = center_x / img_width
    center_y_normalized = center_y / img_height
    width_normalized = width / img_width
    height_normalized = height / img_height

    return [center_x_normalized, center_y_normalized, width_normalized, height_normalized]

   

def convert_to_yolo_obb(corners_pixel, img_width, img_height):
    """
    Konvertiert Pixelkoordinaten von Eckpunkten in das normalisierte
    YOLO OBB Format.

    Args:
        corners_pixel (list or tuple): Eine Liste oder ein Tupel mit 8
            Integer-Werten (x1_px, y1_px, x2_px, y2_px, x3_px, y3_px, x4_px, y4_px).
        img_width (int): Die Breite des Bildes in Pixeln.
        img_height (int): Die Höhe des Bildes in Pixeln.

    Returns:
        tuple: Ein Tupel mit 8 Float-Werten (x1_norm, y1_norm, x2_norm, y2_norm,
               x3_norm, y3_norm, x4_norm, y4_norm) im Bereich [0, 1].
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
    Prüft eine Liste von normierten Werten und wirft einen Fehler,
    wenn ein Wert kleiner als 0 oder größer als 1 ist.

    Args:
        normwerte: Eine Liste von numerischen Werten.

    Raises:
        ValueError: Wenn ein Wert in der Liste kleiner als 0 oder größer als 1 ist.
    """
    found = False
    cp_normalized_values = normalized_values
    for i in range(len(cp_normalized_values)): # Iteriere über die Indizes der Liste
        if cp_normalized_values[i] < 0:
            cp_normalized_values[i] = 0
            found = True
        elif cp_normalized_values[i] > 1:
            cp_normalized_values[i] = 1
            #found = True

    # if found:
    #     print(cp_normalized_values)
    return cp_normalized_values
    #raise ValueError(f"Ungültiger Normwert gefunden: {wert}. Normwerte müssen zwischen 0 und 1 liegen.")
    

def copy_image(source_image_path, destination_folder):
    try:
        with open(source_image_path, 'rb') as source_file:
            image_data = source_file.read()
            filename = os.path.basename(source_image_path)
            destination_image_path = os.path.join(destination_folder, filename)
            with open(destination_image_path, 'wb') as destination_file:
                destination_file.write(image_data)
            #print(f"Image '{filename}' manually copied to '{destination_folder}'.")
    except FileNotFoundError:
        print(f"Error: The source file '{source_image_path}' was not found.")
    except PermissionError:
        print(f"Error: No permission to read the source file or write to the destination folder.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def create_folder_structure(fold_nr):
    def make_directories(path):
        try:
            os.makedirs(path)
            print(f"Ordnerstruktur '{path}' erfolgreich erstellt.")
        except FileExistsError:
            print(f"Einige oder alle Ordner in '{path}' existieren bereits.")
        except Exception as e:
            print(f"Ein Fehler ist aufgetreten: {e}")

    
    path_obj = {
        'path_train_images' : f"code/data/folds/data/fold{fold_nr}/train/images",
        'path_train_labels' : f"code/data/folds/data/fold{fold_nr}/train/labels",
        'path_val_images' : f"code/data/folds/data/fold{fold_nr}/val/images",
        'path_val_labels' : f"code/data/folds/data/fold{fold_nr}/val/labels",
    }
  
    
    make_directories(path_obj['path_train_images'])
    make_directories(path_obj['path_train_labels'])
    
    make_directories(path_obj['path_val_images'])
    make_directories(path_obj['path_val_labels'])
   
    return path_obj


def show_every_picture_with_oriented_bounding_box(path_all_images, path_folds, path_labels, oriented, ir, ret_pts):
    lines_fold = read_file(path_folds)
    labels = read_file(path_labels)
    
   


    for line in lines_fold:
        target = line
        # if ir == True:
        #     image_path = image_path_rgb
        # else:
        #     image_path = image_path_ir
        image_path_rgb = f'{path_all_images}\\{target}_co.png'
        image_path_ir = f'{path_all_images}\\{target}_ir.png'

        filtered_labels = select_all_labels_in_img(target, labels)

        img_rgb = cv2.imread(image_path_rgb)
        img_ir = cv2.imread(image_path_ir)
        img_merged = merge_RGB_and_IR_image(image_path_rgb, image_path_ir, None)

        #transf_labels = transform_labels_to_yolo_format(filtered_labels, img.shape[1], img.shape[0])

        for i in filtered_labels:
            
            img_rgb = calc_pixel_like_authors(img_rgb,i, oriented, ret_pts)
            img_ir = calc_pixel_like_authors(img_ir,i, oriented, ret_pts)
            img_merged = calc_pixel_like_authors(img_merged,i,oriented, ret_pts)
           
            #pts = calc_pixel_like_authors(i)
            


        
        # #cv2.circle(img, pts[0], 2, (0,255,0), 16)
        # cv2.circle(img, pts[1], 2, (0,255,0), 16)
        # cv2.circle(img, pts[2], 2, (0,255,0), 16)
        # cv2.circle(img, pts[3], 2, (0,255,0), 16)
 

        #cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # labels = read_file(label_path)
        # labels = transform_labels_to_yolo_format(labels)
       
        # for i in transf_labels:
        #     img = draw_label_on_image(img, i)
        window_name_rgb = f"{target}"+"_co.png"
        window_name_ir = f"{target}"+"_ir.png"
        window_name_merged = f"{target}"+"_merged.png"
        if img_rgb is not None and img_ir is not None:
            cv2.imshow(window_name_rgb, img_rgb)  # Bild anzeigen
            #cv2.resizeWindow(window_name_rgb, 1024, 1024)
            cv2.imshow(window_name_ir, img_ir)  # Bild anzeigen
            #cv2.resizeWindow(window_name_ir, 1024, 1024)
            cv2.imshow(window_name_merged, img_merged)  # Bild anzeigen
            #cv2.resizeWindow(window_name_merged, 1024, 1024)
            cv2.waitKey(0)  # Warten, bis eine Taste gedrückt wird
            cv2.destroyAllWindows()  # Fenster schließen
      

def draw_axis_aligned_vehicle_bbox(image, Xvehicle, Yvehicle, width_car, length_car, orientationVehicle, veh_type, color=(255, 0, 0), thickness=1):
    """
    Zeichnet die achsenparallele Bounding Box eines Fahrzeugs auf ein OpenCV-Bild.

    Args:
        image (np.ndarray): Das OpenCV-Bild.
        Xvehicle (int): X-Koordinate des Fahrzeugmittelpunkts (Integer für Textpositionierung).
        Yvehicle (int): Y-Koordinate des Fahrzeugmittelpunkts (Integer für Textpositionierung).
        width_car (float): Breite des Fahrzeugs.
        length_car (float): Länge des Fahrzeugs.
        orientationVehicle (float): Orientierung des Fahrzeugs in Radiant.
        veh_type (int): Typ des Fahrzeugs für die Textanzeige.
        color (tuple): Farbe der Bounding Box im BGR-Format (Standard: Blau).
        thickness (int): Dicke der Linien der Bounding Box (Standard: 1).
    """
    cos_theta = np.cos(-orientationVehicle)
    sin_theta = np.sin(-orientationVehicle)

    # Berechne die Eckpunkte des ROTIERTEN Fahrzeugs
    pt1 = np.array([Yvehicle + (width_car / 2) * cos_theta + (length_car / 2) * sin_theta,
                      Xvehicle + (width_car / 2) * sin_theta - (length_car / 2) * cos_theta])
    pt2 = np.array([Yvehicle + (width_car / 2) * cos_theta - (length_car / 2) * sin_theta,
                      Xvehicle + (width_car / 2) * sin_theta + (length_car / 2) * cos_theta])
    pt3 = np.array([Yvehicle - (width_car / 2) * cos_theta - (length_car / 2) * sin_theta,
                      Xvehicle - (width_car / 2) * sin_theta + (length_car / 2) * cos_theta])
    pt4 = np.array([Yvehicle - (width_car / 2) * cos_theta + (length_car / 2) * sin_theta,
                      Xvehicle - (width_car / 2) * sin_theta - (length_car / 2) * cos_theta])

    # Finde die min/max Koordinaten der rotierten Eckpunkte
    all_points = np.array([pt1, pt2, pt3, pt4])
    min_y = int(np.min(all_points[:, 0]))
    max_y = int(np.max(all_points[:, 0]))
    min_x = int(np.min(all_points[:, 1]))
    max_x = int(np.max(all_points[:, 1]))

    # Zeichne die achsenparallele Bounding Box
    bbox_pt1 = (min_x, min_y)
    bbox_pt2 = (max_x, max_y)
    cv2.rectangle(image, bbox_pt1, bbox_pt2, color, thickness)

    label = None
    if veh_type == 1:
        label = 'Car'
    elif veh_type == 2:
        label = 'Truck'
    elif veh_type == 23:
        label = 'Ship'
    elif veh_type == 4:
        label = 'Tractor'
    elif veh_type == 5:
        label = 'Camping Car'
    elif veh_type == 9:
        label = 'van'
    elif veh_type == 10:
        label = 'vehicle'
    elif veh_type == 11:
        label = 'pick-up'
    elif veh_type == 31:
        label = 'plane'

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_x = int(Xvehicle + 15)
        text_y = int(Yvehicle - 15)
        text_bg_color = (0, 0, 0)  # Schwarz
        text_color = (255, 255, 255)  # Weiß
        padding = 2

        # Zeichne den Hintergrund des Textes
        cv2.rectangle(image,
                      (text_x - padding, text_y - text_size[1] - padding),
                      (text_x + text_size[0] + padding, text_y + padding),
                      text_bg_color, cv2.FILLED)

        # Zeichne den Text
        cv2.putText(image, label, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    print("Anzahl Pixel in axis aligned bb")
    print(calculate_bounding_box_area([bbox_pt1, bbox_pt2]))
    return image



def draw_oriented_vehicle_box(image, Xvehicle, Yvehicle, pt1, pt2, pt3, pt4, veh_type, color=(0, 255, 0), thickness=1):
    """
    Zeichnet ein Fahrzeugpolygon und optional einen Textlabel auf ein OpenCV-Bild.

    Args:
        image (np.ndarray): Das OpenCV-Bild, auf das gezeichnet werden soll.
        Xvehicle (int): X-Koordinate des Fahrzeugmittelpunkts (Integer für Textpositionierung).
        Yvehicle (int): Y-Koordinate des Fahrzeugmittelpunkts (Integer für Textpositionierung).
        pt1 (np.ndarray): Koordinaten des ersten Eckpunkts [y, x].
        pt2 (np.ndarray): Koordinaten des zweiten Eckpunkts [y, x].
        pt3 (np.ndarray): Koordinaten des dritten Eckpunkts [y, x].
        pt4 (np.ndarray): Koordinaten des vierten Eckpunkts [y, x].
        veh_type (int): Typ des Fahrzeugs für die Textanzeige.
        color (tuple): Farbe des Polygons im BGR-Format (Standard: Grün).
        thickness (int): Dicke der Polygonlinien (Standard: 2).
    """
    # Konvertiere die Punktkoordinaten in Integer-Tupel für cv2.line
    p1 = (int(pt1[1]), int(pt1[0]))
    p2 = (int(pt2[1]), int(pt2[0]))
    p3 = (int(pt3[1]), int(pt3[0]))
    p4 = (int(pt4[1]), int(pt4[0]))

    # Zeichne die Linien des Polygons
    cv2.line(image, p1, p2, color, thickness)
    cv2.line(image, p2, p3, color, thickness)
    cv2.line(image, p3, p4, color, thickness)
    cv2.line(image, p4, p1, color, thickness)

    print("Anzahl der pixel in einer oriented bb sind: " + str(calculate_oriented_bounding_box_area([p1,p2,p3,p4])))
  
    if veh_type is not None:
        label = f"{veh_type}"
    else:
        label = None
    if veh_type == '001':
        label = 'Car'
    elif veh_type == '002':
        label = 'Truck'
    elif veh_type == '023':
        label = 'Ship'
    elif veh_type == '004':
        label = 'Tractor'
    elif veh_type == '005':
        label = 'Camping Car'
    elif veh_type == '009':
        label = 'van'
    elif veh_type == '010':
        label = 'vehicle'
    elif veh_type == '011':
        label = 'pick-up'
    elif veh_type == '031':
        label = 'plane'

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_x = int(Xvehicle + 15)
        text_y = int(Yvehicle - 15)
        text_bg_color = (0, 0, 0)  # Schwarz
        text_color = (255, 255, 255)  # Weiß
        padding = 2

        # Zeichne den Hintergrund des Textes
        cv2.rectangle(image,
                      (text_x - padding, text_y - text_size[1] - padding),
                      (text_x + text_size[0] + padding, text_y + padding),
                      text_bg_color, cv2.FILLED)

        # Zeichne den Text
        cv2.putText(image, label, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return image


def calc_pixel_like_authors(img, label, oriented, ret_pts):
    indice_vehicle = label.split()
    Xvehicle = np.float64(indice_vehicle[1])
    Yvehicle = np.float64(indice_vehicle[2])
    orientationVehicle = indice_vehicle[3]


    veh_type = indice_vehicle[12]
    x = np.array(indice_vehicle[4:8], np.float32)
    y = np.array(indice_vehicle[8:12], np.float32)
   

  
    l_1 = np.sqrt((x[0]-x[1])**2 + (y[0]-y[1])**2)
    l_2 = np.sqrt((x[1]-x[2])**2 + (y[1]-y[2])**2)
    l_3 = np.sqrt((x[2]-x[3])**2 + (y[2]-y[3])**2)
    l_4 = np.sqrt((x[3]-x[0])**2 + (y[3]-y[0])**2)
    l = np.array([l_1,l_2,l_3,l_4])

    ktri = np.argsort(l)[::-1] #key triangle
    ltri = l[ktri]              # längen Triangle

    length_car = np.mean(ltri[:2])
    width_car = np.mean(ltri[2:])

    index1_start = ktri[0]
    index1_end = (ktri[0] + 1) % 4
    index2_start = ktri[1]
    index2_end = (ktri[1] + 1) % 4
    vector_length1 = np.array([x[index1_end] - x[index1_start], y[index1_end] - y[index1_start]])
    vector_length2 = np.array([x[index2_end] - x[index2_start], y[index2_end] - y[index2_start]])

    if np.dot(vector_length1, vector_length2) > 0:
        vector_length = (vector_length1 + vector_length2) / 2
    else:
        vector_length = (vector_length1 - vector_length2) / 2

    if vector_length[0] == 0:
        orientationVehicle = -np.pi / 2
    else:
        orientationVehicle = np.arctan(vector_length[1] / vector_length[0])


    cos_theta = np.cos(-orientationVehicle)
    sin_theta = np.sin(-orientationVehicle)

    pt1 = np.array([
        Yvehicle + (width_car / 2) * cos_theta + (length_car / 2) * sin_theta,
        Xvehicle + (width_car / 2) * sin_theta - (length_car / 2) * cos_theta
    ])

    pt2 = np.array([
        Yvehicle + (width_car / 2) * cos_theta - (length_car / 2) * sin_theta,
        Xvehicle + (width_car / 2) * sin_theta + (length_car / 2) * cos_theta
    ])

    pt3 = np.array([
        Yvehicle - (width_car / 2) * cos_theta - (length_car / 2) * sin_theta,
        Xvehicle - (width_car / 2) * sin_theta + (length_car / 2) * cos_theta
    ])

    pt4 = np.array([
        Yvehicle - (width_car / 2) * cos_theta + (length_car / 2) * sin_theta,
        Xvehicle - (width_car / 2) * sin_theta - (length_car / 2) * cos_theta
    ])


    p1 = np.array(pt1, np.int64)
    p2 = np.array(pt2, np.int64)
    p3 = np.array(pt3, np.int64)
    p4 = np.array(pt4, np.int64)
    pts = [p1, p2, p3, p4]


    if ret_pts == True and oriented == True :
        p1 = (int(pt1[0]), int(pt1[1]))
        p2 = (int(pt2[0]), int(pt2[1]))
        p3 = (int(pt3[0]), int(pt3[1]))
        p4 = (int(pt4[0]), int(pt4[1]))
        return [veh_type,p1,p2,p3,p4]
    elif ret_pts == True and oriented == False:
        cos_theta = np.cos(-orientationVehicle)
        sin_theta = np.sin(-orientationVehicle)

        # Berechne die Eckpunkte des ROTIERTEN Fahrzeugs
        pt1 = np.array([Yvehicle + (width_car / 2) * cos_theta + (length_car / 2) * sin_theta,
                    Xvehicle + (width_car / 2) * sin_theta - (length_car / 2) * cos_theta])
        pt2 = np.array([Yvehicle + (width_car / 2) * cos_theta - (length_car / 2) * sin_theta,
                    Xvehicle + (width_car / 2) * sin_theta + (length_car / 2) * cos_theta])
        pt3 = np.array([Yvehicle - (width_car / 2) * cos_theta - (length_car / 2) * sin_theta,
                    Xvehicle - (width_car / 2) * sin_theta + (length_car / 2) * cos_theta])
        pt4 = np.array([Yvehicle - (width_car / 2) * cos_theta + (length_car / 2) * sin_theta,
                    Xvehicle - (width_car / 2) * sin_theta - (length_car / 2) * cos_theta])

        # Finde die min/max Koordinaten der rotierten Eckpunkte
        all_points = np.array([pt1, pt2, pt3, pt4])
        min_y = int(np.min(all_points[:, 0]))
        max_y = int(np.max(all_points[:, 0]))
        min_x = int(np.min(all_points[:, 1]))
        max_x = int(np.max(all_points[:, 1]))

        # Zeichne die achsenparallele Bounding Box
        bbox_pt1 = [min_x, min_y]
        bbox_pt2 = [max_x, max_y]

        p1 = [bbox_pt1[1], bbox_pt1[0]]
        p2 = [bbox_pt2[1], bbox_pt2[0]]
        p3 = [min_y, max_x]
        p4 = [max_y, min_x]
        return [veh_type,p1,p2,p3,p4]
    elif oriented == True:
        return draw_oriented_vehicle_box(img,Xvehicle,Yvehicle,pt1,pt2,pt3,pt4,veh_type,(255,0,0),2)
    else:
        return draw_axis_aligned_vehicle_bbox(img,Xvehicle,Yvehicle,width_car,length_car,orientationVehicle,veh_type)
    
    

def calculate_bounding_box_area(points):
    """
    Berechnet die Fläche (Anzahl der Pixel) der achsenparallelen Bounding Box
    um eine Menge von Punkten.

    Args:
        points (list or np.ndarray): Eine Liste von Tupeln oder ein NumPy-Array
                                      der Form (n, 2) mit den (y, x)
                                      oder (x, y) Koordinaten der Punkte. Die Funktion
                                      versucht, die richtige Ordnung zu erkennen,
                                      nimmt aber standardmäßig (y, x) an.

    Returns:
        int: Die Anzahl der Pixel innerhalb der Bounding Box. Gibt 0 zurück,
             wenn keine Punkte vorhanden sind.
    """
    if points is None or len(points) < 1:
        return 0
    print(points)
    points_np = np.array(points)

    # Annahme: Punkte sind im Format [y, x]. Wenn nicht, transponieren wir.
    # Überprüfen anhand des Mittelwerts, welche Spalte tendenziell größere Werte hat (oft x).
    if points_np.shape[1] == 2 and np.mean(points_np[:, 1]) > np.mean(points_np[:, 0]):
        points_np = points_np[:, [1, 0]]  # Spalten tauschen: [x, y] -> [y, x]

    if points_np.shape[1] != 2:
        raise ValueError("Die Eingabepunkte müssen die Form (n, 2) haben.")

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
    Berechnet die Fläche (Anzahl der Pixel) der orientierten Bounding Box
    um eine Menge von Punkten.

    Args:
        points (list or np.ndarray): Eine Liste von Tupeln oder ein NumPy-Array
                                      der Form (n, 2) mit den (y, x)
                                      oder (x, y) Koordinaten der Punkte.

    Returns:
        float: Die Fläche der orientierten Bounding Box in Pixeleinheiten.
               Gibt 0.0 zurück, wenn keine oder weniger als 3 Punkte vorhanden sind.
    """
    points_np = np.array(points)
    if points_np.shape[0] < 3:
        return 0.0

    if points_np.shape[1] != 2:
        raise ValueError("Die Eingabepunkte müssen die Form (n, 2) haben.")

    # Finde die konvexe Hülle der Punkte
    hull = ConvexHull(points_np)
    hull_points = points_np[hull.vertices]

    # Finde die orientierte Bounding Box mit OpenCV
    rect = cv2.minAreaRect(hull_points.astype(np.float32))
    width, height = rect[1]
    area = width * height
    return area


def select_all_labels_in_img(target, labels):
    # Filtert alle Labels, die mit den Anfangsziffern von 'target' übereinstimmen
    filtered_labels = [label for label in labels if label.startswith(target)]
    return filtered_labels

 
def transform_labels_to_yolo_format(labels, width, height):
    yolo_labels = []  # Liste für die konvertierten YOLO-Labels
    for label in labels:
        label_parts = label.split()  # Teilt das Label in Teile basierend auf Leerzeichen
        if len(label_parts) >= 13:  # Sicherstellen, dass genügend Teile vorhanden sind
            try:
                class_id = int(label_parts[0])  # Konvertiere die Klasse in eine Ganzzahl
                x_center = float(label_parts[1])  # Konvertiere x_center in eine Gleitkommazahl
                y_center = float(label_parts[2])  # Konvertiere y_center in eine Gleitkommazahl
              
                x1 = int(label_parts[4])
                y1 = int(label_parts[5])
                x2 = int(label_parts[6])
                y2 = int(label_parts[7])
                x3 = int(label_parts[8])
                y3 = int(label_parts[9])
                x4 = int(label_parts[10])
                y4 = int(label_parts[11])

                # Konvertiere die Teile in das YOLO-Format
                yolo_label = convert_to_yolo_classic(
                    x_center, y_center, x1, y1, x2, y2, x3, y3, x4, y4, class_id, width, height
                )
                yolo_labels.append(yolo_label)
            except ValueError as e:
                print(f"Fehler beim Verarbeiten des Labels: {label} - {e}")
        else:
            print(f"Ungültiges Label-Format: {label}")
    return yolo_labels

def convert_to_yolo_classic(x_center, y_center, x1, y1, x2, y2, x3, y3, x4, y4, class_id, img_width, img_height):
    x_min = min(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    x_max = max(x1, x2, x3, x4)
    y_max = max(y1, y2, y3, y4)

    width = x_max - x_min
    height = y_max - y_min

    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    return f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"


def create_yaml(path):
    file_path = f"{path}/data.yaml"
    print(file_path)
    file_string = "train: ./train/images " +'\n'+ "val: ./val/images" +'\n'+  "nc: 9"  +'\n'+"names: ['Car', 'Truck', 'Ship', 'Tractor', 'Camping Car', 'van', 'vehicle', 'pick-up', 'plane']"

    try:
        with open(file_path, 'w') as file:
            file.write(file_string)
        print("YAML erfolgreich erstellt")
    except IOError as e:
        print(f"Fehler beim Schreiben der YAML-Datei: {e}")

def read_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file]  # Entfernt \n aus allen Zeilen
    return lines
main()