from pathlib import Path
import numpy as np
import os
from preproc_folds import read_file, select_all_labels_in_img

def main():
    """
    Main routine for creating custom folds from a list of images and their class counts.

    - Sets configuration flags and output paths.
    - Loads all image filenames and corresponding label data.
    - Creates a list of image-class objects.
    - Calls create_own_folds to split images into folds and writes results to disk.
    """

    bools_object = {
        "palma":False,
    }

    output_path = r'Code\data\folds\own_folds'



    if bools_object['palma'] == True:
        path_all_images = r'../../../scratch/tmp/t_liet02/all_vedai_images'
        path_labels = r'../../../scratch/tmp/t_liet02/annotations/annotation.txt'
        
    else:
        path_all_images = r'Code\data\all_vedai_images'
        path_labels = r'Code\data\annotation.txt'

    list_all_files = list_files(path_all_images)
    print(len(list_files(path_all_images)))

    image_class_list = create_image_class_object(list_all_files, path_labels)

    create_own_folds(5, image_class_list, output_path)



def create_own_folds(number_of_folds, image_class_list, output_path):
    """
    Splits images into a specified number of folds, balancing object counts across folds.

    Args:
        number_of_folds (int): Number of folds to create.
        image_class_list (list): List of dictionaries with image IDs and object counts.
        output_path (str): Directory to write fold files and statistics.

    Returns:
        int: Always returns 0 after processing.
    """

    number_of_folds += 1
    img_counter = 0
    print(number_of_folds)

    fold_arr = [[] for _ in range(number_of_folds)]
    fold_arr_object_counter = [[] for _ in range(number_of_folds)]

    np.random.shuffle(image_class_list)

    for i in range(number_of_folds):
        fold_arr[i].append(image_class_list[i]['image_id'])
    img_counter += number_of_folds
    print("fold_arr")
    print(fold_arr)
    fold_arr_object_counter = count_objects_in_fold_arr(fold_arr, image_class_list)
    #print(fold_arr_object_counter)


    i = img_counter
    error_counter =0
    empty_images = []
    print( len(image_class_list))
    while img_counter != len(image_class_list):
        for i in image_class_list:
            #print(i)
            smallest_classes = get_smallest_class_in_image(i["image_id"], image_class_list)
            #print(smallest_classes)
            if len(smallest_classes) == 1:
                smallest_fold = get_indices_of_folds_with_smallest_object_count(fold_arr_object_counter, smallest_classes[0])
                fold_arr[smallest_fold[0]].append(i['image_id'])
                
                fold_arr_object_counter = count_objects_in_fold_arr(fold_arr, image_class_list)
            elif len(smallest_classes) >= 2:
                #Reihenfolge nach Haeufigkeit: Car, Pick-up, Camping-Car, Truck, Vehicle, Tractor, Ship, van, plane
                freq_val_arr=[]
                for j in smallest_classes:
                    if j == "car":
                        freq_val = 0
                    elif j == "pick-up":
                        freq_val = 1   
                    elif j == "camping_car":
                        freq_val = 2   
                    elif j == "truck":
                        freq_val = 3 
                    elif j == "vehicle":
                        freq_val = 4   
                    elif j == "tractor":
                        freq_val = 5
                    elif j == "ship":
                        freq_val = 6
                    elif j == "van":
                        freq_val = 7
                    elif j == "plane":
                        freq_val = 8
                    freq_val_arr.append([freq_val, j])       
                max_tupel = max(freq_val_arr, key=lambda x: x[0])

                smallest_fold = get_indices_of_folds_with_smallest_object_count(fold_arr_object_counter, max_tupel[1])
                fold_arr[smallest_fold[0]].append(i['image_id'])
                
                fold_arr_object_counter = count_objects_in_fold_arr(fold_arr, image_class_list)
            else: #image is empty
                error_counter +=1
                
                empty_images.append(i['image_id'])
              
            
            
            
            img_counter += 1
            print("Image " + str(img_counter) + "/" + str(len(image_class_list)))

            if img_counter == len(image_class_list):
                break
            
 

    print("error_counter")
    print(error_counter)
    print("__")
    print(empty_images)
    number_empty_images = np.zeros(number_of_folds, dtype=int)


    quotient, rest = divmod(error_counter, number_of_folds)

    index = 0
    for i in range(len(fold_arr)):
        for _ in range(quotient):
            fold_arr[i].append(empty_images.pop())
            number_empty_images[i] += 1

    while empty_images:
        # Bestimme das Fold mit der kleinsten Laenge
        min_index = min(range(len(fold_arr)), key=lambda i: len(fold_arr[i]) if isinstance(fold_arr[i], list) else float('inf'))

        # Fuege das naechste Element aus empty_images hinzu
        fold_arr[min_index].append(empty_images.pop())  # oder pop() fuer schnellere Variante
        number_empty_images[min_index] += 1




        
    object_counter = count_objects_in_fold_arr(fold_arr, image_class_list)
    counter = 0
    fold_image_count_array =  []
    for i in fold_arr:
       
        print("images fold " + str(counter) + ": "+str(len(i)))
        print(i[0:5])
        print(object_counter[counter])
        fold_image_count_array.append(len(i))
        counter += 1
        print("___")


    write_folds(fold_arr, object_counter, fold_image_count_array, output_path, error_counter, number_empty_images)
      
            

                


  




    return 0

def write_folds(arr, object_counter, fold_image_count_arr, output_dir, error_counter, number_empty_images):
    """
    Writes the fold assignments and statistics to text files.

    Args:
        arr (list): List of lists, each containing image IDs for a fold.
        object_counter (list): List of dictionaries with object counts per fold.
        fold_image_count_arr (list): List of image counts per fold.
        output_dir (str): Directory to write output files.
        error_counter (int): Number of images with no objects.
        number_empty_images (np.ndarray): Array with number of empty images per fold.
    """
    print("write data")
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(arr)):
        img_list = arr[i]
        filename = os.path.join(output_dir, f"fold{i}.txt")
        with open(filename, "w") as f:
            for line in img_list:
                f.write(line + "\n")
    extra_filename = os.path.join(output_dir, "statistics.txt")
    line_counter = 0
    with open(extra_filename, "w") as f:
        for line in object_counter:
            f.write(str(line) +" Images quantity: "+str(fold_image_count_arr[line_counter])+ " / Number of Background Images:  " + str(number_empty_images[line_counter]) +  "\n")
            line_counter += 1
        f.write("Images with no objects: " + str(error_counter) )
        print("write" + str(i))

    print("finished writing")


def get_indices_of_folds_with_smallest_object_count(fold_arr_counter, object_name):
    """
    Finds the indices of folds with the smallest count of a given object.

    Args:
        fold_arr_counter (list): List of dictionaries with object counts per fold.
        object_name (str): Name of the object to count (e.g., "car").

    Returns:
        list: List of indices of folds with the smallest count for the specified object.
    """
    if not fold_arr_counter:
        return []

    min_count = float('inf')
    # Speichern der Indizes der Folds mit der kleinsten Zaehlung
    smallest_fold_indices = []

    for i, fold_data in enumerate(fold_arr_counter):
        if object_name in fold_data["data"]:
            current_count = fold_data["data"][object_name]

            if current_count < min_count:
                min_count = current_count
                smallest_fold_indices = [i] # Starte eine neue Liste, da ein kleinerer Wert gefunden wurde
            elif current_count == min_count:
                smallest_fold_indices.append(i) # Fuege Index hinzu, wenn der Wert gleich ist

    return smallest_fold_indices


def get_smallest_class_in_image(i, image_class_list):
    """
    Finds the class(es) with the smallest count in a given image.

    Args:
        i (str): Image ID.
        image_class_list (list): List of dictionaries with image IDs and object counts.

    Returns:
        list: List of class names with the smallest count in the image.
    """
    result = next((item for item in image_class_list if item["image_id"] == i), None)
    if result is None:
        return None

    data = {k: v for k, v in result["data"].items() if v > 0}

    if not data:  # Handle case where all values are 0 or data is empty
        return []

    # Find the minimum value
    min_value = min(data.values())

    # Find all classes that have this minimum value
    smallest_classes = [k for k, v in data.items() if v == min_value]

    return smallest_classes
    
def count_objects_in_fold_arr(fold_arr, image_class_list):
    """
    Counts the number of objects per class in each fold.

    Args:
        fold_arr (list): List of lists, each containing image IDs for a fold.
        image_class_list (list): List of dictionaries with image IDs and object counts.

    Returns:
        list: List of dictionaries with object counts per fold.
    """
    fold_arr_counter = []
    for i in range(len(fold_arr)):
        fold_counter = {
            "fold": f"{i+1}",
            "data": {
                "car": 0,
                "truck": 0,
                "ship": 0,
                "tractor": 0,
                "camping_car": 0,
                "van": 0,
                "vehicle": 0,
                "pick-up": 0,
                "plane": 0,
            }
        }
        fold_arr_counter.append(fold_counter)
    for i, inner in enumerate(fold_arr):
        for element in inner:
            result = next((item for item in image_class_list if item['image_id'] == element), None)
            if result:
                for key in fold_arr_counter[i]["data"]:
                    fold_arr_counter[i]["data"][key] += result['data'][key]

    
    return fold_arr_counter

def create_image_class_object(list_all_files, path_labels):
    """
    Creates a list of image-class objects by counting objects in each image.

    Args:
        list_all_files (list): List of image filenames.
        path_labels (str): Path to the label file.

    Returns:
        list: List of dictionaries with image IDs and object counts.
    """

    labels = read_file(path_labels)
    image_class_list = []

    for i in list_all_files:
        filtered_labels = select_all_labels_in_img(i, labels)
        counted_objects = count_objects(filtered_labels)
        temp = {
            "image_id":i,
            "data": counted_objects
        }
        image_class_list.append(temp)
    return image_class_list


def count_objects(filtered_labels):
    """
    Counts the number of objects per class in a list of label strings.

    Args:
        filtered_labels (list): List of label strings for one image.

    Returns:
        dict: Dictionary with object counts for each class.
    """
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

    object_counter = {
        "car":0,
        "truck":0,
        "ship":0,
        "tractor":0,
        "camping_car":0,
        "van":0,
        "vehicle":0,
        "pick-up":0,
        "plane":0,
    }
    for i in filtered_labels:
        splitted_string = i.split()
        if len(splitted_string) < 13:
            continue  # skip malformed lines
        class_id = splitted_string[12]
        yolo_class = convert_class_to_yolo(class_id)
        if yolo_class == '0':
            object_counter['car'] += 1
        elif yolo_class == '1':
            object_counter['truck'] += 1
        elif yolo_class == '2':
            object_counter['ship'] += 1
        elif yolo_class == '3':
            object_counter['tractor'] += 1
        elif yolo_class == '4':
            object_counter['camping_car'] += 1
        elif yolo_class == '5':
            object_counter['van'] += 1
        elif yolo_class == '6':
            object_counter['vehicle'] += 1
        elif yolo_class == '7':
            object_counter['pick-up'] += 1
        elif yolo_class == '8':
            object_counter['plane'] += 1
    return object_counter


def list_files(folder_path):
    """
    Lists all image files in a folder, removing file endings.

    Args:
        folder_path (str): Path to the folder containing image files.

    Returns:
        list: List of image IDs without file endings.
    """
    all_images = [f.name for f in Path(folder_path).iterdir() if f.is_file()]
    all_images_without_ending = list(set(i[:-7] if len(i) >= 7 else '' for i in all_images))
    return all_images_without_ending

if __name__ == "__main__":
    main()


