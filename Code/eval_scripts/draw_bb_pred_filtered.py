import json
import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from preproc_folds import get_bounding_box_in_px, read_file, select_all_labels_in_img



# Klassen + Farben (BGR für OpenCV) OLD


CLASS_MAP = {  # BGR
    1: ("Car",        (44, 160, 44)),     # grün
    2: ("Truck",      (40, 39, 214)),     # rot
    3: ("Ship",       (180, 119, 31)),    # blau
    4: ("Tractor",    (14, 127, 255)),    # Orange
    5: ("CampingCar", (189, 103, 148)),   # lila
    6: ("Van",        (75, 86, 140)),     # braun
    7: ("Vehicle",    (194, 119, 227)),   # pink
    8: ("Pick-up",    (127, 255, 0)),     # Türkis/Grünblau (gut kontrastierend zu Truck & Plane)
    9: ("Plane",      (34, 189, 188)),    # Hellblau
    10: ("Ground Truth", (0, 0, 0))       # schwarz
}
def draw_predictions(input_path, json_path, output_path, image_id, score_threshold=0.3, allowed_classes=None, all_images=True):
    """
    Zeichnet Vorhersagen in ein Bild.
    
    :param json_path: Pfad zur JSON-Datei mit den Predictions
    :param output_path: Ausgabeort für gespeichertes Bild
    :param image_id: ID des Bildes (ohne Endung)
    :param score_threshold: minimaler Score
    :param allowed_classes: Liste von Klassennamen (z.B. ["Car", "Truck"]) oder None für alle
    """
    pred_name = input_path[0]
    fold_num = input_path[1]
    image_id_path = str(image_id).zfill(8)
    if all_images:
        path_all_images = r'Code\data\all_vedai_images'
        base = Path(path_all_images)
        image_path = base / f"{image_id_path}_co.png"
        print(image_path)
    else:
        base = r"Code\data\testfold_data"
        image_path = rf"{base}\{pred_name}\images\{image_id_path}.png"
        print(image_path)


    




    image_id_raw = image_id
    image_id = int(image_id)

    base_output = r"MA-Thesis-Latex/images/015Results"
    output_path_full = rf"{base_output}\{output_path}\comp_images\{pred_name}\{image_id}.png"
    
    with open(json_path, "r") as f:
        predictions = json.load(f)

    # Bild laden
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Bild {image_path} konnte nicht geladen werden.")

    # Filtere nur Predictions für das gewünschte Bild und über Schwellwert
    preds_for_image = [
        p for p in predictions if p["image_id"] == image_id and p["score"] >= score_threshold
    ]

    # ggf. nach Klassen filtern
    # if allowed_classes is not None:
    #     preds_for_image = [
    #         p for p in preds_for_image
    #         if CLASS_MAP.get(p["category_id"], ("", ""))[0] in allowed_classes
    #     ]

    # # Wenn keine passende Prediction → Bild überspringen
    # if len(preds_for_image) == 0:
    #     return False  

    for pred in preds_for_image:
        cat_id = pred["category_id"]
        label, color = CLASS_MAP.get(cat_id, (f"cat{cat_id}", (200, 200, 200)))

        poly = pred["poly"]
        points = np.array(poly, dtype=np.int32).reshape((-1, 2))

        # Dünne Bounding Box einzeichnen
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)

        # # Text vorbereiten
        # text = f"{label} {pred['score']:.2f}"
        # x, y = points[0]
        # font = cv2.FONT_HERSHEY_PLAIN
        # font_scale = 1
        # font_thickness = 1

        # # Textgröße bestimmen
        # (w, h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

        # # Halbtransparenten Hintergrund hinter den Text zeichnen
        # overlay = image.copy()
        # cv2.rectangle(overlay, (x, y - h - 4), (x + w, y), color, -1)
        # alpha = 0.5
        # cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # # Text darauf schreiben (weiß)
        # cv2.putText(image, text, (x, y - 2), font, font_scale, (255, 255, 255), font_thickness)
         # Text vorbereiten
        text = f"{label} {pred['score']:.2f}"
        text = f"{pred['score']:.2f}"
        x, y = points[0]
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1
        font_thickness = 1

        # Textgröße bestimmen
        (w, h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

        # Schwarzen Hintergrund hinter den Text zeichnen
        cv2.rectangle(image, (x, y - h - 4), (x + w, y), (0, 0, 0), -1)

        # Text darauf schreiben (weiß)
        cv2.putText(image, text, (x, y - 2), font, font_scale, (255, 255, 255), font_thickness)

    # Bild anzeigen
    print("Saving to:", output_path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

   
    success = cv2.imwrite(str(output_path_full), image)
    print("Saved:", success)
    # cv2.imshow("Predictions: " + str(image_id_raw), image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return True  # Bild wurde angezeigt



def generate_ground_truth(results_base,output_path, images):
    path_labels = r'Code\data\annotation.txt'
    base_output = r"MA-Thesis-Latex/images/015Results"
    output_path_full = rf"{base_output}\{output_path}\comp_images\ground_truth"

    labels = read_file(path_labels)
    for _, image_id in images:
        print(image_id)
        image_id_zeros = str(image_id).zfill(8)
        filtered_labels = select_all_labels_in_img(image_id_zeros, labels)
        base = r"Code\data\testfold_data"
        image_path = rf"{base}\ir\images\{image_id_zeros}.png"
        img = cv2.imread(image_path)

        for i in filtered_labels:
            img = get_bounding_box_in_px(img,i, True, False)

        # window_name_rgb = f"{image_id_zeros}"+".png"
        # cv2.imshow(window_name_rgb, img)
        # cv2.waitKey(0)  # Warten, bis eine Taste gedrÃ¼ckt wird
        # cv2.destroyAllWindows() 
        output_path_with_png = rf"{base_output}\{output_path}\comp_images\ground_truth\{image_id}.png"
        cv2.imwrite(output_path_with_png, img)







    return 0
if __name__ == "__main__":
    output_path = "03ablation"
    pred_arr = [
        ("red", 2),
        ("green", 3),
        ("blue", 2),
        ("ir", 0),
        ("ndvi", 0)
    ]

    images = [
        ("car", 523),
        ("truck", 212),
        ("ship", 509),
        ("tractor", 523),
        ("van", 198),
        ("pick up", 523),
        ("plane", 487),
        ("vehicle", 427),
        ("camping car", 523)
    ]
    
    results_base = Path(r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\master_thesis\MA-Thesis-Latex\images\015Results\03ablation\comp_images")
    generate_ground_truth(results_base, output_path,images)

    for pred_name, fold_num in pred_arr:
        # Dynamischer JSON-Pfad
        json_path = rf"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\new_val_detect\{pred_name}\fold{fold_num}\val_at_test\predictions.json"

        for _, image_id in images:
            
            input_path = [pred_name, fold_num]
            draw_predictions(
                input_path,
                json_path=json_path,
                image_id=image_id,
                output_path=output_path,
                score_threshold=0.3,
                all_images=False                
            )
