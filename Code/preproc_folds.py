import cv2
import numpy as np
from scipy.spatial import ConvexHull

def main():
    oriented = True
    path_folds = r'Code\data\folds\txts\fold01.txt'
    path_all_images = r'Code\data\all_vedai_images'
    path_labels = r'Code\data\all_vedai_images\annotation.txt'
   
    lines_fold = read_file(path_folds)
    
    show_every_picture_with_oriented_bounding_box(path_all_images,path_folds,path_labels,oriented)

    #create_folds(path_all_images,path_folds,path_labels)

def create_folds(path_all_images,path_folds,path_labels):

    return 0
def show_every_picture_with_oriented_bounding_box(path_all_images, path_folds, path_labels, oriented, ir):
    lines_fold = read_file(path_folds)
    labels = read_file(path_labels)
    print(labels[0])

    for line in lines_fold:
        target = line
        if ir == True:
            image_path = f"{path_all_images}\{target}_ir.png"
        else:
            image_path = f"{path_all_images}\{target}_co.png"

        filtered_labels = select_all_labels_in_img(target, labels)
        print(filtered_labels)

        img = cv2.imread(image_path)

        transf_labels = transform_labels_to_yolo_format(filtered_labels, img.shape[1], img.shape[0])

        for i in filtered_labels:
            
            img = calc_pixel_like_authors(img,i, oriented)
            #pts = calc_pixel_like_authors(i)
            


        
        # #cv2.circle(img, pts[0], 2, (0,255,0), 16)
        # cv2.circle(img, pts[1], 2, (0,255,0), 16)
        # cv2.circle(img, pts[2], 2, (0,255,0), 16)
        # cv2.circle(img, pts[3], 2, (0,255,0), 16)
 

        #cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # labels = read_file(label_path)
        # labels = transform_labels_to_yolo_format(labels)
        print(transf_labels)
        # for i in transf_labels:
        #     img = draw_label_on_image(img, i)

        if img is not None:
            cv2.imshow("Gefundene PNG-Datei", img)  # Bild anzeigen
            cv2.waitKey(0)  # Warten, bis eine Taste gedrückt wird
            cv2.destroyAllWindows()  # Fenster schließen
        print(target)

def draw_axis_aligned_vehicle_bbox(image, Xvehicle, Yvehicle, width_car, length_car, orientationVehicle, veh_type, color=(255, 0, 0), thickness=2):
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
        thickness (int): Dicke der Linien der Bounding Box (Standard: 2).
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



def draw_oriented_vehicle_box(image, Xvehicle, Yvehicle, pt1, pt2, pt3, pt4, veh_type, color=(0, 255, 0), thickness=2):
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
    print(veh_type)
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
    print("x1")
    if label:
        print("t1")
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




def calc_pixel_like_authors(img, label, oriented):
    indice_vehicle = label.split()
    Xvehicle = np.float64(indice_vehicle[1])
    Yvehicle = np.float64(indice_vehicle[2])
    orientationVehicle = indice_vehicle[3]


    veh_type = indice_vehicle[12]
    x = np.array(indice_vehicle[4:8], np.int32)
    y = np.array(indice_vehicle[8:12], np.int32)
    print(x)
    print(y)

  
    l_1 = np.sqrt((x[0]-x[1])**2 + (y[0]-y[1])**2)
    l_2 = np.sqrt((x[1]-x[2])**2 + (y[1]-y[2])**2)
    l_3 = np.sqrt((x[2]-x[3])**2 + (y[2]-y[3])**2)
    l_4 = np.sqrt((x[3]-x[0])**2 + (y[3]-y[0])**2)
    l = np.array([l_1,l_2,l_3,l_4])
    print(l)
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

    print(label)

    p1 = np.array(pt1, np.int64)
    p2 = np.array(pt2, np.int64)
    p3 = np.array(pt3, np.int64)
    p4 = np.array(pt4, np.int64)
    pts = [p1, p2, p3, p4]

    if oriented == True:
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
                print("label parts "+str(label_parts[2]))
                x1 = int(label_parts[4])
                y1 = int(label_parts[5])
                x2 = int(label_parts[6])
                y2 = int(label_parts[7])
                x3 = int(label_parts[8])
                y3 = int(label_parts[9])
                x4 = int(label_parts[10])
                y4 = int(label_parts[11])

                # Konvertiere die Teile in das YOLO-Format
                yolo_label = convert_to_yolo(
                    class_id, x_center, y_center, x1, y1, x2, y2, x3, y3, x4, y4, class_id, width, height
                )
                yolo_labels.append(yolo_label)
            except ValueError as e:
                print(f"Fehler beim Verarbeiten des Labels: {label} - {e}")
        else:
            print(f"Ungültiges Label-Format: {label}")
    return yolo_labels

def convert_to_yolo(image_id, x_center, y_center, x1, y1, x2, y2, x3, y3, x4, y4, class_id, img_width, img_height):
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


def read_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file]  # Entfernt \n aus allen Zeilen
    return lines
main()