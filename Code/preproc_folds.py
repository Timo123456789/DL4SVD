import cv2
import numpy as np
from scipy.spatial import ConvexHull
import os

def main():
    """
    Main function to configure and execute various preprocessing tasks for VEDAI dataset images and annotations.
    This function sets up parameters for image and label paths, dataset options, and feature selection.
    Depending on the configuration, it can display images with oriented bounding boxes, create cross-validation folds,
    generate permutation datasets, or perform ablation studies. The function supports toggling infrared and NDVI channels,
    merging IR data, and switching between local and remote (Palma) paths.
    Key operations (some commented out) include:
    - Displaying images with oriented bounding boxes.
    - Creating cross-validation folds with specified features.
    - Generating permutation and ablation datasets.
    Parameters are set within the function and not passed as arguments.
    Note:
    - Modify the boolean flags and parameters to select the desired preprocessing operation.
    - Ensure the required data files and directories exist at the specified paths.
    """
    oriented = True
    ir = False
    bool_create_yaml = True
    limiter = None
    palma = False
    merge_ir_bool= False
    namestring = ""


    perm_object={"r":True,
                 "g":True,
                 "b":True,
                 "ir":False,
                 "ndvi":False,
                 }

    if palma == True:
        path_all_images = r'../../../scratch/tmp/t_liet02/all_vedai_images'
        path_labels = r'../../../scratch/tmp/t_liet02/annotations/annotation.txt'
        
    else:
        path_all_images = r'Code\data\all_vedai_images'
        path_labels = r'Code\data\annotation.txt'
        
    show_every_picture_with_oriented_bounding_box(path_all_images, r'Code\data\folds\tempfold\fold5.txt', path_labels, True, False, False, perm_object)
    #create_aab_oob_cross_method(path_all_images, path_labels, ir, bool_create_yaml, limiter, merge_ir_bool, namestring, palma)

    #create_perm_dataset(path_all_images, path_labels, ir, bool_create_yaml, limiter, merge_ir_bool, namestring, palma, perm_object)


    #create_ablation_datasets(path_all_images, path_labels, ir, bool_create_yaml, limiter, merge_ir_bool, namestring, palma)
    merge_ir_bool=True
    oriented = True
    path_fold_dest_string = r'data/cross_validation/rgbndvi'
    perm_object={"r":True,
                 "g":True,
                 "b":True,
                 "ir":False,
                 "ndvi":True,
                 }
    
  
    #create_fold_cross_validation(0,path_all_images,path_labels, ir, True, bool_create_yaml, limiter, oriented, merge_ir_bool, namestring, palma, path_fold_dest_string, perm_object)
    print_divideline()

def create_ablation_datasets(path_all_images, path_labels, ir, bool_create_yaml, limiter, merge_ir_bool, namestring, palma):
    """
    Creates ablation datasets for cross-validation by generating dataset folds with different channel permutations.
    This function generates cross-validation datasets where each dataset contains only one of the image channels 
    (red, green, blue, infrared, or NDVI) enabled at a time. For each channel, five folds are created using the 
    `create_fold_cross_validation` function. The datasets are saved in corresponding directories.
    Args:
        path_all_images (str): Path to the directory containing all images.
        path_labels (str): Path to the directory containing label files.
        ir (Any): Infrared channel information or configuration.
        bool_create_yaml (bool): Whether to create a YAML file for each fold.
        limiter (Any): Limiting parameter for dataset creation (e.g., number of samples).
        merge_ir_bool (bool): Whether to merge infrared channel data.
        namestring (str): String used for naming or identification.
        palma (Any): Additional parameter for dataset creation (purpose context-dependent).
    Returns:
        int: Returns 0 upon completion.
    """
    oriented=True
    merge_ir_bool = True
    # path_fold_dest_string = r'data/cross_validation_ablation/red'
    # perm_object={"r":True,
    #              "g":False,
    #              "b":False,
    #              "ir":False,
    #              "ndvi":False,
    #              }
    # for fold_nr in range(5):  
    #     create_fold_cross_validation(fold_nr,path_all_images,path_labels, ir, True, bool_create_yaml, limiter, oriented, merge_ir_bool, namestring, palma, path_fold_dest_string, perm_object)

    # path_fold_dest_string = r'data/cross_validation_ablation/green'
    # perm_object={"r":False,
    #              "g":True,
    #              "b":False,
    #              "ir":False,
    #              "ndvi":False,
    #              }
    # for fold_nr in range(5):  
    #     create_fold_cross_validation(fold_nr,path_all_images,path_labels, ir, True, bool_create_yaml, limiter, oriented, merge_ir_bool, namestring, palma, path_fold_dest_string, perm_object)


    # path_fold_dest_string = r'data/cross_validation_ablation/blue'
    # perm_object={"r":False,
    #              "g":False,
    #              "b":True,
    #              "ir":False,
    #              "ndvi":False,
    #              }
    # for fold_nr in range(5):  
    #     create_fold_cross_validation(fold_nr,path_all_images,path_labels, ir, True, bool_create_yaml, limiter, oriented, merge_ir_bool, namestring, palma, path_fold_dest_string, perm_object)


    # path_fold_dest_string = r'data/cross_validation_ablation/ir'
    # perm_object={"r":False,
    #              "g":False,
    #              "b":False,
    #              "ir":True,
    #              "ndvi":False,
    #              }
    # for fold_nr in range(5):  
    #     create_fold_cross_validation(fold_nr,path_all_images,path_labels, ir, True, bool_create_yaml, limiter, oriented, merge_ir_bool, namestring, palma, path_fold_dest_string, perm_object)

    path_fold_dest_string = r'data/cross_validation_ablation/ndvi'
    perm_object={"r":False,
                 "g":False,
                 "b":False,
                 "ir":False,
                 "ndvi":True,
                 }
    for fold_nr in range(5):  
        create_fold_cross_validation(fold_nr,path_all_images,path_labels, ir, True, bool_create_yaml, limiter, oriented, merge_ir_bool, namestring, palma, path_fold_dest_string, perm_object)
    return 0



def create_aab_oob_cross_method(path_all_images, path_labels, ir, bool_create_yaml, limiter, merge_ir_bool, namestring, palma):
    """
    Creates cross-validation folds using the out-of-bag (OOB) method for image data.
    This function generates five cross-validation folds by calling `create_fold_cross_validation` for each fold.
    The folds are created using the provided image and label paths, along with various configuration parameters.
    Args:
        path_all_images (str): Path to the directory containing all images.
        path_labels (str): Path to the labels corresponding to the images.
        ir (Any): Infrared-related parameter passed to the fold creation function.
        bool_create_yaml (bool): If True, YAML files are created for each fold.
        limiter (Any): Parameter to limit the number of samples or other aspects during fold creation.
        merge_ir_bool (bool): If True, infrared data is merged during fold creation.
        namestring (str): A string used for naming or identification purposes.
        palma (Any): Additional parameter for fold creation, possibly related to data or configuration.
    Returns:
        int: Returns 0 upon completion.
    """

    path_fold_dest_string = r'data/cross_validation/obb'
   
    oriented=True
    fold_nr = 0
    for fold_nr in range(5):  
        create_fold_cross_validation(fold_nr,path_all_images,path_labels, ir, True, bool_create_yaml, limiter, oriented, merge_ir_bool, namestring, palma, path_fold_dest_string, None)

    #path_fold_dest_string = r'data/cross_validation/obb'
    #fold_nr = 0
    #oriented=True
    #for fold_nr in range(5):  
    #    create_fold_cross_validation(fold_nr,path_all_images,path_labels, ir, True, bool_create_yaml, limiter, oriented, merge_ir_bool, namestring, palma, path_fold_dest_string, None)
            
    return 0
            


def create_perm_dataset(path_all_images, path_labels, ir, bool_create_yaml, limiter, merge_ir_bool, namestring, palma, perm_object):
    """
    Creates multiple cross-validation dataset folds with different channel permutations.
    This function generates datasets for cross-validation by varying the inclusion of image channels
    (e.g., RGB, IR, NDVI) according to predefined permutations. For each permutation, it calls
    `create_fold_cross_validation` for five folds and organizes the output in corresponding directories.
    Args:
        path_all_images (str): Path to the directory containing all images.
        path_labels (str): Path to the labels file or directory.
        ir (Any): Infrared channel data or configuration.
        bool_create_yaml (bool): Whether to create YAML configuration files for each fold.
        limiter (Any): Limiting parameter for dataset creation (e.g., number of samples).
        merge_ir_bool (bool): Whether to merge IR channel with other channels.
        namestring (str): Name identifier for the dataset or experiment.
        palma (Any): Additional configuration or parameter for dataset creation.
        perm_object (dict): Dictionary specifying which channels to include (keys: 'r', 'g', 'b', 'ir', 'ndvi').
    Returns:
        int: Returns 0 upon completion.
    """
    merge_ir_bool=True
    oriented = True
    path_fold_dest_string = r'data/cross_validation/rgbir'
    perm_object={"r":True,
                 "g":True,
                 "b":True,
                 "ir":True,
                 "ndvi":False,
                 }
    
    fold_nr = 0
    for fold_nr in range(5):  
        create_fold_cross_validation(fold_nr,path_all_images,path_labels, ir, True, bool_create_yaml, limiter, oriented, merge_ir_bool, namestring, palma, path_fold_dest_string, perm_object)
    print_divideline()

    # path_fold_dest_string = r'data/cross_validation/irgb'
    # perm_object={"r":False,
    #              "g":True,
    #              "b":True,
    #              "ir":True,
    #              "ndvi":False,
    #              }
    # fold_nr = 0
    # for fold_nr in range(5):  
    #     create_fold_cross_validation(fold_nr,path_all_images,path_labels, ir, True, bool_create_yaml, limiter, oriented, merge_ir_bool, namestring, palma, path_fold_dest_string, perm_object)
    # print_divideline()

    # path_fold_dest_string = r'data/cross_validation/rirb'
    # perm_object={"r":True,
    #              "g":False,
    #              "b":True,
    #              "ir":True,
    #              "ndvi":False,
    #              }
    # fold_nr = 0
    # for fold_nr in range(5):  
    #     create_fold_cross_validation(fold_nr,path_all_images,path_labels, ir, True, bool_create_yaml, limiter, oriented, merge_ir_bool, namestring, palma, path_fold_dest_string, perm_object)
    # print_divideline()

    # path_fold_dest_string = r'data/cross_validation/rgir'
    # perm_object={"r":True,
    #              "g":True,
    #              "b":False,
    #              "ir":True,
    #              "ndvi":False,
    #              }
    # fold_nr = 0
    # for fold_nr in range(5):  
    #     create_fold_cross_validation(fold_nr,path_all_images,path_labels, ir, True, bool_create_yaml, limiter, oriented, merge_ir_bool, namestring, palma, path_fold_dest_string, perm_object)
    # print_divideline()

    path_fold_dest_string = r'data/cross_validation/rgbndvi'
    perm_object={"r":True,
                 "g":True,
                 "b":True,
                 "ir":False,
                 "ndvi":True,
                 }
    fold_nr = 0
    for fold_nr in range(5):  
        create_fold_cross_validation(fold_nr,path_all_images,path_labels, ir, True, bool_create_yaml, limiter, oriented, merge_ir_bool, namestring, palma, path_fold_dest_string, perm_object)
    print_divideline()

    # path_fold_dest_string = r'data/cross_validation/gbndvi'
    # perm_object={"r":False,
    #              "g":True,
    #              "b":True,
    #              "ir":False,
    #              "ndvi":True,
    #              }
    # fold_nr = 0
    # for fold_nr in range(5):  
    #     create_fold_cross_validation(fold_nr,path_all_images,path_labels, ir, True, bool_create_yaml, limiter, oriented, merge_ir_bool, namestring, palma, path_fold_dest_string, perm_object)
    # print_divideline()



    return 0






def create_all_folds(path_all_images, path_labels, ir, oriented,bool_create_yaml, limiter, merge_ir_bool, namestring, palma, path_fold_dest_string, perm_object):
    """
    Creates multiple data folds for cross-validation or similar purposes.
    Iterates through a predefined number of folds (12), and for each fold (except the first and last),
    calls the `create_fold` function with appropriate parameters. For the 10th fold, a specific flag is set.
    Args:
        path_all_images (str): Path to the directory containing all images.
        path_labels (str): Path to the labels file or directory.
        ir (Any): Infrared-related parameter, passed to `create_fold`.
        oriented (Any): Orientation-related parameter, passed to `create_fold`.
        bool_create_yaml (bool): Whether to create a YAML file for each fold.
        limiter (Any): Parameter to limit the number of samples or other aspects.
        merge_ir_bool (bool): Whether to merge infrared data.
        namestring (str): String used for naming or identification.
        palma (Any): Additional parameter, possibly related to processing or configuration.
        path_fold_dest_string (str): Destination path for the created folds.
        perm_object (Any): Permutation or randomization object for fold creation.
    Returns:
        None
    """
    fold_nr = 1


    for fold_nr in range(12):
        if fold_nr != 0 and fold_nr < 11:
            if fold_nr == 10:
                create_fold(fold_nr, path_all_images,path_labels, ir, True,bool_create_yaml, limiter, oriented, merge_ir_bool, namestring, palma, path_fold_dest_string, perm_object)
            else:
                create_fold(fold_nr, path_all_images,path_labels, ir, False,bool_create_yaml, limiter, oriented, merge_ir_bool, namestring, palma, path_fold_dest_string, perm_object)
          

        #except:
        #    print("No fold number found")




def merge_RGB_and_IR_image(rgb_path, ir_path, perm_object):
    """
    Merges an RGB image and an infrared (IR) image into a multi-channel image, optionally including an NDVI channel.
    Args:
        rgb_path (str): Path to the RGB image file.
        ir_path (str): Path to the IR image file (expected as a single-band grayscale image).
        perm_object (dict): Dictionary specifying which channels to include in the output.
            Keys should include 'r', 'g', 'b', 'ir', and optionally 'ndvi' (boolean values).
    Returns:
        numpy.ndarray or None: The merged image containing the selected channels as specified in perm_object.
            Returns None if images cannot be loaded or if their dimensions do not match.
    Notes:
        - The function expects both images to have the same height and width.
        - The merged image may have 2 or more channels, depending on perm_object.
        - If 'ndvi' is True in perm_object, the NDVI channel is calculated and included.
        - Visualization of the resulting multi-channel image may require custom handling.
    """
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
        channel_data = {
            "r": r,
            "g": g,
            "b": b,
            "ir": ir_img,
            "ndvi": None,
        }
        if perm_object is not None and perm_object["ndvi"] is True:
            channel_data["ndvi"] = calc_ndvi(r, ir_img)
        channel_order = ["r", "g", "b", "ir", "ndvi"]
        
        
        active_channels = [channel_data[key] for key in channel_order if perm_object.get(key)]
        

        # Jetzt kannst du mit cv2.merge arbeiten
        if len(active_channels) >= 2:
            merged_img = cv2.merge(active_channels)
        elif len(active_channels) == 1:
            merged_img = active_channels[0]
        else:
            print("Mindestens zwei aktive KanÃ¤le nÃ¶tig fÃ¼r cv2.merge()")

        # The resulting image now has 4 channels: Blue, Green, Red, Infrared.
        # Note that the interpretation and visualization of such a
        # 4-channel image depend on the subsequent processing.
        # Standard image viewers usually expect RGB or RGBA.

        # Save the image with the added IR band (e.g., as PNG, which supports alpha)
        # Even though we call it 'merged_img', it's essentially a 4-channel image.
        #cv2.imwrite('rgb_ir_merged.png', merged_img)
        #print("RGB image with IR band successfully saved.")

        return merged_img

        # Displaying the image is not trivial as it has 4 channels that cannot be
        # directly interpreted as RGB. You would need to decide how to combine
        # these 4 channels for visualization (e.g., mapping IR to a color channel).
        # Here, we only show it as a raw 4-channel array (which might not look meaningful).
        # cv2.imshow('RGB + IR', merged_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

def calc_ndvi(r,ir):
    """
    Calculates the Normalized Difference Vegetation Index (NDVI) from red and infrared bands.

    Parameters
    ----------
    r : np.ndarray
        The red band of the image as a NumPy array.
    ir : np.ndarray
        The infrared band of the image as a NumPy array.

    Returns
    -------
    np.ndarray
        The NDVI values scaled to the range [0, 255] as an unsigned 8-bit integer array.

    Notes
    -----
    NDVI is computed as (ir - r) / (ir + r). To avoid division by zero, a small value (0.01)
    is added where the denominator is zero. The result is scaled to [0, 255] for visualization.
    """
    r = r.astype(np.float32)
    ir = ir.astype(np.float32)
    bottom = (ir + r)
    bottom[bottom == 0] = 0.01
    ndvi = (ir - r) / bottom
    return (ndvi*255).astype(np.uint8)



def create_fold(
    fold_nr,
    path_all_images,
    path_labels,
    ir,
    fold10bool,
    bool_create_yaml,
    limiter,
    oriented,
    merge_ir_bool,
    namestring,
    palma_bool,
    fold_dest_path,
    perm_object
):
    """
    Creates a data fold for training and validation by organizing images and labels into the appropriate directory structure,
    copying and processing images, generating label files, and optionally creating a YAML configuration file.
    Args:
        fold_nr (int): The fold number to process.
        path_all_images (str): Path to the directory containing all images.
        path_labels (str): Path to the file containing all labels.
        ir (bool): Whether to include infrared images.
        fold10bool (bool): If True, use the 10th fold configuration; otherwise, use the specified fold number.
        bool_create_yaml (bool): If True, create a YAML configuration file for the fold.
        limiter (int or None): Limits the number of images processed for training and validation. If None, processes all images.
        oriented (bool): Whether to use oriented bounding boxes for labels.
        merge_ir_bool (bool): If True, merge infrared images with color images.
        namestring (str): A string tag to identify the fold.
        palma_bool (bool): If True, use Palma-specific paths and structure.
        fold_dest_path (str): Destination path for the fold's output.
        perm_object (object): An object containing permission or configuration information for image copying.
    Returns:
        int: Returns 0 upon successful creation of the fold.
    Side Effects:
        - Copies images and creates label files in the specified directory structure.
        - Optionally creates a YAML configuration file.
        - Prints progress and status messages to the console.
    """
    def create_image_and_label(lines,  string_tag):
        """
        Processes a list of image identifiers to create corresponding image and label files for a specified dataset split.
        Args:
            lines (list): List of image identifiers (strings) to process.
            string_tag (str): Tag indicating the dataset split (e.g., 'train', 'val').
        Side Effects:
            - Copies image files to the target directory.
            - Creates label files for each image.
            - Prints progress information to the console.
            - May break early if a limiter is set for the number of images to process.
        Notes:
            - Uses global variables: path_all_images, labels, paths_object, merge_ir_bool, perm_object, ir, oriented, fold_nr, limiter.
            - For 'train', stops after 'limiter' images if limiter is set.
            - For 'val', stops after ceil(limiter/10) images if limiter is set.
        """
        counter = 0
        for line in lines:
            counter += 1
            target = line
          
            image_path = f"{path_all_images}/{target}_co.png"         
            image_path_ir = f"{path_all_images}/{target}_ir.png"

            filtered_labels = select_all_labels_in_img(target, labels)
            copy_image(image_path, paths_object['path_'+string_tag+'_images'], merge_ir_bool, image_path, image_path_ir, perm_object)
            create_label_file(target, filtered_labels, paths_object['path_'+string_tag+'_labels'], image_path, ir, oriented, string_tag)
            print("Fold Nr:"+str(fold_nr)+" /  "+ string_tag+" image: "+str(counter) + "/" + str(len(lines)))
            
            if limiter != None:
                if counter == limiter and string_tag == 'train':
                    print("Limiter! BREAK "+ string_tag+" at " + str(limiter))
                    break
                elif counter == (np.ceil(limiter/10)) and string_tag == 'val':
                    print("Limiter! BREAK "+ string_tag+" at " + str(limiter))
                    break
            

    if fold10bool == True and palma_bool == True:
        fold_train_images_path = rf'../../../scratch/tmp/t_liet02/folds/txts/fold10.txt'
        fold_val_images_path = rf'../../../scratch/tmp/t_liet02/folds/txts/fold10test.txt'
        yaml_path = rf'../../../scratch/tmp/t_liet02/{fold_dest_path}/fold10{namestring}'
    elif fold10bool == True and palma_bool == False:
        fold_train_images_path = rf'Code\data\folds\txts\fold10.txt'
        fold_val_images_path = rf'Code\data\folds\txts\fold10test.txt'
        yaml_path = rf'Code\data\folds\{fold_dest_path}\fold10'
    elif fold10bool == False and palma_bool == True:
        fold_train_images_path = rf'../../../scratch/tmp/t_liet02/folds/txts/fold0{fold_nr}.txt'
        fold_val_images_path = rf'../../../scratch/tmp/t_liet02/folds/txts/fold0{fold_nr}test.txt'
        yaml_path = rf'../../../scratch/tmp/t_liet02/{fold_dest_path}/fold{fold_nr}{namestring}'
    elif fold10bool == False and palma_bool == False:
        fold_train_images_path = rf'Code\data\folds\txts\fold0{fold_nr}.txt'
        fold_val_images_path = rf'Code\data\folds\txts\fold0{fold_nr}test.txt'
        yaml_path = rf'Code\data\folds\{fold_dest_path}\fold{fold_nr}'

    

    paths_object = create_folder_structure(fold_nr,namestring, palma_bool, fold_dest_path)

    lines_fold_train = read_file(fold_train_images_path)
    lines_fold_val = read_file(fold_val_images_path)
    labels = read_file(path_labels)
    
    if bool_create_yaml:
        create_yaml(yaml_path, fold_nr, namestring, palma_bool, fold_dest_path)
        print("Yaml successfull created.")
    

    create_image_and_label(lines_fold_train, "train")
    create_image_and_label(lines_fold_val, "val")
   
    print("Fold " + str(fold_nr) + " / " + namestring + " in ' "+fold_dest_path+" ' with val nr "+str(fold_nr)+" successfull created.")
    return 0




   


def create_label_file(target, labels, path, img_path, ir, oriented, string_tag):
    """
    Creates a YOLO-format label file for a given image and its associated labels.
    Args:
        target (str): The base name for the output label file.
        labels (list): A list of label data, each containing class and bounding box information.
        path (str): Directory path where the label file will be saved.
        img_path (str): Path to the image file corresponding to the labels.
        ir (bool): Indicates whether the image is infrared (currently unused).
        oriented (bool): If True, uses oriented bounding boxes (OBB) format; otherwise, uses classic YOLO format.
        string_tag (str): Tag indicating the dataset split (e.g., "train", "val", "test").
    Raises:
        Exception: If an error occurs during file writing or label processing.
    Notes:
        - The function reads the image to obtain its dimensions for bounding box normalization.
        - Bounding boxes are converted to YOLO format (classic or OBB) depending on the 'oriented' flag.
        - The output file is named '{target}.txt' and saved in the specified 'path'.
        - Class IDs are mapped to YOLO class indices using an internal mapping.
    """
    def convert_class_to_yolo(class_id):
        """
        Converts a given class ID string to its corresponding YOLO class index as a string.

        Args:
            class_id (str): The class ID to convert. Expected values are:
                '001' - Car
                '002' - Truck
                '023' - Ship
                '004' - Tractor
                '005' - Camping Car
                '009' - van
                '010' - vehicle
                '011' - pick-up
                '031' - plane

        Returns:
            str: The YOLO class index as a string. If the class_id does not match any known value,
                 returns '6' (vehicle) by default.
        """
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

    # if ir == True:
    #     file_path = f"{path}/{target}_ir.txt"
    # else:
    #     file_path = f"{path}/{target}_co.txt"

    file_path = f"{path}/{target}.txt"

    img = cv2.imread(img_path)

    
  
    try:
        with open(file_path, 'w') as file:
            for label in labels:
                transf_label = get_bounding_box_in_px(img, label, oriented, True) 
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
                    #UrsprÃ¼ngliches YOLO Format


                    yolo_transf_label = convert_label_to_yolo_classic(px_corner,img.shape[1], img.shape[0])
                    file_string = (convert_class_to_yolo(transf_label[0]) + " "+
                       str(yolo_transf_label[0])+" "+
                       str(yolo_transf_label[1])+" "+
                       str(yolo_transf_label[2])+" "+ 
                       str(yolo_transf_label[3]) + '\n')

                    #obb YOLO Format aber keine obb boxen
                    # yolo_transf_label = convert_to_yolo_abb(px_corner, img.shape[1], img.shape[0])
                    # file_string = (convert_class_to_yolo(transf_label[0]) + " "+
                    #     str(yolo_transf_label[0])+" "+
                    #     str(yolo_transf_label[1])+" "+
                    #     str(yolo_transf_label[2])+" "+ 
                    #     str(yolo_transf_label[3])+" "+ 
                    #     str(yolo_transf_label[4])+" "+ 
                    #     str(yolo_transf_label[5])+" "+
                    #     str(yolo_transf_label[6])+" "+
                    #     str(yolo_transf_label[7]) + '\n')
                    
                file.write(file_string)
            #datei.write(inhalt)
        #print(f"Die Datei '{file_path}' wurde erfolgreich mit Inhalt erstellt.")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten in der create_label_file: {e}")

def convert_to_yolo_abb(corners_pixel, img_width, img_height):
    """
    Converts four corner points to YOLOv9 OBB label format.

    Args:
        corners_pixel (list): [x1, y1, x2, y2, x3, y3, x4, y4] in pixels
        img_width (int): Image width in pixels
        img_height (int): Image height in pixels

    Returns:
        str: YOLOv9 OBB label with normalized coordinates
                'class_id x1 y1 x2 y2 x3 y3 x4 y4'
    """

    normalized_points = []
    for i in range(0, 8, 2):
        x_norm = corners_pixel[i] / img_width
        y_norm = corners_pixel[i+1] / img_height
        normalized_points.extend([x_norm, y_norm])

    return check_normalvalues(normalized_points)


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

    return check_normalvalues([center_x_normalized, center_y_normalized, width_normalized, height_normalized])


def convert_to_yolo_obb(corners_pixel, img_width, img_height):
    """
    Converts the pixel coordinates of the four corners of an oriented bounding box (OBB)
    to normalized coordinates suitable for YOLO OBB format.
    Args:
        corners_pixel (list or tuple): A sequence of 8 values representing the pixel coordinates
            of the four corners of the bounding box in the order:
            [x1, y1, x2, y2, x3, y3, x4, y4].
        img_width (int or float): The width of the image in pixels.
        img_height (int or float): The height of the image in pixels.
    Returns:
        list: A list of 8 normalized values corresponding to the input corners,
            checked and possibly adjusted by the `check_normalvalues` function.
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

    norm_values = check_normalvalues([x1_norm, y1_norm, x2_norm, y2_norm, x3_norm, y3_norm, x4_norm, y4_norm])

    return norm_values


def check_normalvalues(normalized_values):
    """
    Ensures that all values in the input list are within the range [0, 1].
    If a value is less than 0, it is set to 0. If a value is greater than 1, it is set to 1.
    Returns the modified list.
    Args:
        normalized_values (list or array-like): List of numeric values to check and correct.
    Returns:
        list: List with all values clipped to the range [0, 1].
    """
    
    found = False
    cp_normalized_values = normalized_values
    for i in range(len(cp_normalized_values)): # Iteriere Ã¼ber die Indizes der Liste
        if cp_normalized_values[i] < 0:
            cp_normalized_values[i] = 0
            found = True
        elif cp_normalized_values[i] > 1:
            cp_normalized_values[i] = 1
            #found = True

    # if found:
    #     print(cp_normalized_values)
    return cp_normalized_values
    #raise ValueError(f"UngÃ¼ltiger Normwert gefunden: {wert}. Normwerte mÃ¼ssen zwischen 0 und 1 liegen.")
    

def copy_image(source_image_path, destination_folder, merge_ir_bool, image_path_rgb, image_path_ir, perm_object):
    """
    Copies an image from the source path to the destination folder, optionally merging RGB and IR images.
    If `merge_ir_bool` is True and `perm_object` is provided, merges the RGB and IR images using the specified
    channel permissions and saves the result as a TIFF file if the number of channels exceeds 3. Otherwise,
    copies the source image as-is.
    The destination filename is constructed from the first part of the original filename (up to 8 characters)
    and retains the original extension unless merging with more than 3 channels, in which case '.tiff' is used.
    Args:
        source_image_path (str): Path to the source image file.
        destination_folder (str): Path to the destination folder where the image will be copied.
        merge_ir_bool (bool): Whether to merge RGB and IR images.
        image_path_rgb (str): Path to the RGB image (used if merging).
        image_path_ir (str): Path to the IR image (used if merging).
        perm_object (dict or None): Dictionary specifying which channels to include in the merge. If None, defaults to 3 channels.
    Raises:
        FileNotFoundError: If the source image file does not exist.
        PermissionError: If there is no permission to read the source or write to the destination.
        Exception: For any other unexpected errors.
    """
    if perm_object is not None:
        number_of_channels = sum(1 for v in perm_object.values() if v)
    else:
        number_of_channels = 3
    try:
        if merge_ir_bool == True and perm_object is not None:
            with open(source_image_path, 'rb') as source_file:
                image_data = merge_RGB_and_IR_image(image_path_rgb, image_path_ir, perm_object)
                filename = os.path.basename(source_image_path)

                parts = filename.split('_')
                base_name_part = parts[0]
                _, extension = os.path.splitext(filename)
                if number_of_channels > 3:
                    extension=".tiff"
                filename = f"{base_name_part[:8]}{extension}"


                destination_image_path = os.path.join(destination_folder, filename)
                cv2.imwrite(destination_image_path, image_data)
        else:
            with open(source_image_path, 'rb') as source_file:
                image_data = source_file.read()
                filename = os.path.basename(source_image_path)

                parts = filename.split('_')
                base_name_part = parts[0]
                _, extension = os.path.splitext(filename)
                filename = f"{base_name_part[:8]}{extension}"

                destination_image_path = os.path.join(destination_folder, filename)
                with open(destination_image_path, 'wb') as destination_file:
                    #destination_file.write(image_data)
                    destination_file.write(image_data)
                #print(f"Image '{filename}' manually copied to '{destination_folder}'.")
    except FileNotFoundError:
        print(f"Error: The source file '{source_image_path}' was not found.")
    except PermissionError:
        print(f"Error: No permission to read the source file or write to the destination folder.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def create_folder_structure(fold_nr, namestring, palma_bool, dest_path, test_dataset_bool):
    """
    Creates a folder structure for training, validation, and optionally test datasets, 
    based on the specified parameters and returns the paths as a dictionary.
    Args:
        fold_nr (int): The fold number to include in the folder names.
        namestring (str): Additional string to append to the folder names.
        palma_bool (bool): If True, use Palma cluster paths; if False, use local paths.
        dest_path (str): Destination path segment to include in the folder structure.
        test_dataset_bool (bool): If True, include test dataset folders; otherwise, exclude them.
    Returns:
        dict: A dictionary containing the paths for train, val, and optionally test images and labels.
    Side Effects:
        Creates the specified directories on the filesystem. Prints status messages for each directory.
    """
    def make_directories(path):
        try:
            os.makedirs(path)
            print(f"Ordnerstruktur '{path}' erfolgreich erstellt.")
        except FileExistsError:
            print(f"Einige oder alle Ordner in '{path}' existieren bereits.")
        except Exception as e:
            print(f"Ein Fehler ist aufgetreten: {e}")

    if palma_bool == True and test_dataset_bool == False:
        path_obj = {
            'path_train_images' : f"../../../scratch/tmp/t_liet02/{dest_path}/fold{fold_nr}{namestring}/train/images",
            'path_train_labels' : f"../../../scratch/tmp/t_liet02/{dest_path}/fold{fold_nr}{namestring}/train/labels",
            'path_val_images' : f"../../../scratch/tmp/t_liet02/{dest_path}/fold{fold_nr}{namestring}/val/images",
            'path_val_labels' : f"../../../scratch/tmp/t_liet02/{dest_path}/fold{fold_nr}{namestring}/val/labels",
            # 'path_test_images' : f"../../../scratch/tmp/t_liet02/{dest_path}/fold{fold_nr}{namestring}/test/images",
            # 'path_test_labels' : f"../../../scratch/tmp/t_liet02/{dest_path}/fold{fold_nr}{namestring}/test/labels",
            }
    elif palma_bool == True and test_dataset_bool == True:
        path_obj = {
            'path_train_images' : f"../../../scratch/tmp/t_liet02/{dest_path}/fold{fold_nr}{namestring}/train/images",
            'path_train_labels' : f"../../../scratch/tmp/t_liet02/{dest_path}/fold{fold_nr}{namestring}/train/labels",
            'path_val_images' : f"../../../scratch/tmp/t_liet02/{dest_path}/fold{fold_nr}{namestring}/val/images",
            'path_val_labels' : f"../../../scratch/tmp/t_liet02/{dest_path}/fold{fold_nr}{namestring}/val/labels",
            'path_test_images' : f"../../../scratch/tmp/t_liet02/{dest_path}/fold{fold_nr}{namestring}/test/images",
            'path_test_labels' : f"../../../scratch/tmp/t_liet02/{dest_path}/fold{fold_nr}{namestring}/test/labels",
            }
    elif palma_bool == False and test_dataset_bool == True:
        path_obj = {
            'path_train_images' : f"code/data/folds/{dest_path}/fold{fold_nr}/train/images",
            'path_train_labels' : f"code/data/folds/{dest_path}/fold{fold_nr}/train/labels",
            'path_val_images' : f"code/data/folds/{dest_path}/fold{fold_nr}/val/images",
            'path_val_labels' : f"code/data/folds/{dest_path}/fold{fold_nr}/val/labels",
            'path_test_images' : f"code/data/folds/{dest_path}/fold{fold_nr}/test/images",
            'path_test_labels' : f"code/data/folds/{dest_path}/fold{fold_nr}/test/labels",
        }
    elif palma_bool == False and test_dataset_bool == False:
        path_obj = {
            'path_train_images' : f"code/data/folds/{dest_path}/fold{fold_nr}/train/images",
            'path_train_labels' : f"code/data/folds/{dest_path}/fold{fold_nr}/train/labels",
            'path_val_images' : f"code/data/folds/{dest_path}/fold{fold_nr}/val/images",
            'path_val_labels' : f"code/data/folds/{dest_path}/fold{fold_nr}/val/labels",
            # 'path_test_images' : f"code/data/folds/{dest_path}/fold{fold_nr}/test/images",
            # 'path_test_labels' : f"code/data/folds/{dest_path}/fold{fold_nr}/test/labels",
        }
  
    
    make_directories(path_obj['path_train_images'])
    make_directories(path_obj['path_train_labels'])
    
    make_directories(path_obj['path_val_images'])
    make_directories(path_obj['path_val_labels'])

    if test_dataset_bool:
        make_directories(path_obj['path_test_images'])
        make_directories(path_obj['path_test_labels'])
   
    return path_obj


def show_every_picture_with_oriented_bounding_box(path_all_images, path_folds, path_labels, oriented, ir, ret_pts, perm_object):
    """
    Displays every image specified in the fold file with oriented bounding boxes drawn on them.
    For each image listed in the fold file, this function:
    - Loads the corresponding RGB and IR images.
    - Merges RGB and IR images if required.
    - Selects all labels for the current image.
    - Draws oriented bounding boxes on the RGB, IR, and merged images for each label.
    - Optionally prints label information.
    - Saves the processed RGB image to a temporary location.
    - Optionally splits and saves color channels as grayscale images.
    - Displays the RGB image in a resizable window.
    Args:
        path_all_images (str): Path to the directory containing all images.
        path_folds (str): Path to the file listing image identifiers for the current fold.
        path_labels (str): Path to the file containing label information.
        oriented (bool): Whether to draw oriented bounding boxes.
        ir (bool): If True, use IR images; otherwise, use RGB images.
        ret_pts (bool): If True, return bounding box corner points.
        perm_object (Any): Permutation object used for merging RGB and IR images.
    Returns:
        None
    """
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
        img_merged = merge_RGB_and_IR_image(image_path_rgb, image_path_ir, perm_object)

        #transf_labels = transform_labels_to_yolo_format(filtered_labels, img.shape[1], img.shape[0])

        for i in filtered_labels:
            
            img_rgb = get_bounding_box_in_px(img_rgb,i, oriented, ret_pts)
            img_ir = get_bounding_box_in_px(img_ir,i, oriented, ret_pts)
            img_merged = get_bounding_box_in_px(img_merged,i,oriented, ret_pts)
            print(label_as_txt(i))
           
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
        cv2.imwrite(r"Code\data\folds\tempfold\output_image_smaller0.png", img_rgb)

        if img_rgb is not None:
    # Kanäle extrahieren
            b, g, r = cv2.split(img_rgb)
            # Speichern als Graustufenbilder
            #cv2.imwrite(r"Code\data\folds\tempfold\output_image_blue.png", b)
            #cv2.imwrite(r"Code\data\folds\tempfold\output_image_green.png", g)
            #cv2.imwrite(r"Code\data\folds\tempfold\output_image_red.png", r)
            #cv2.imwrite(r"Code\data\folds\tempfold\output_image_schrott_ir.png", img_ir)
        if img_rgb is not None and img_ir is not None:
            cv2.imshow(window_name_rgb, img_rgb)  # Bild anzeigen
         
            cv2.resizeWindow(window_name_rgb, 1000, 1000)  # Breite=800px, Höhe=600px

         
            #cv2.resizeWindow(window_name_rgb, 1024, 1024)
            #cv2.imshow(window_name_ir, img_ir)  # Bild anzeigen
            #cv2.resizeWindow(window_name_ir, 1024, 1024)
            #cv2.imshow(window_name_merged, img_merged)  # Bild anzeigen
            #cv2.resizeWindow(window_name_merged, 1024, 1024)
            cv2.waitKey(0)  # Warten, bis eine Taste gedrÃ¼ckt wird
            cv2.destroyAllWindows()  # Fenster schlieÃŸen


def label_as_txt(string):
    """
    Converts a vehicle type code extracted from a space-separated string into a human-readable label.

    Args:
        string (str): A space-separated string where the vehicle type code is expected at index 12.

    Returns:
        str or None: The corresponding vehicle label if the code matches known types, otherwise the code itself or None.
        Known vehicle type codes:
            '001' -> 'Car'
            '002' -> 'Truck'
            '023' -> 'Ship'
            '004' -> 'Tractor'
            '005' -> 'Camping Car'
            '009' -> 'van'
            '010' -> 'vehicle'
            '011' -> 'pick-up'
            '031' -> 'plane'
    """
    parts = string.split()
    veh_type = parts[12]
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
    return label

def draw_axis_aligned_vehicle_bbox(image, Xvehicle, Yvehicle, width_car, length_car, orientationVehicle, veh_type, color=(255, 0, 0), thickness=2):
    """
    Draws an axis-aligned bounding box around a rotated vehicle on the given image and annotates it with its type.
    The function computes the rotated corners of the vehicle based on its position, size, and orientation,
    then determines the axis-aligned bounding box that encloses these corners. It draws this bounding box
    on the image and adds a label indicating the vehicle type.
    Args:
        image (np.ndarray): The image on which to draw the bounding box.
        Xvehicle (float): The x-coordinate (column) of the vehicle's center.
        Yvehicle (float): The y-coordinate (row) of the vehicle's center.
        width_car (float): The width of the vehicle.
        length_car (float): The length of the vehicle.
        orientationVehicle (float): The orientation angle of the vehicle in radians.
        veh_type (int): The type of the vehicle (used for labeling).
        color (tuple, optional): The color of the bounding box in BGR format. Default is (255, 0, 0).
        thickness (int, optional): The thickness of the bounding box lines. Default is 2.
    Returns:
        np.ndarray: The image with the drawn bounding box and label.
    Notes:
        - The function prints the area of the axis-aligned bounding box using `calculate_bounding_box_area`.
        - Supported vehicle types and their labels include:
            1: 'Car'
            2: 'Truck'
            23: 'Ship'
            4: 'Tractor'
            5: 'Camping Car'
            9: 'van'
            10: 'vehicle'
            11: 'pick-up'
            31: 'plane'
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
    color = (0, 0, 0)
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
        text_color = (255, 255, 255)  # WeiÃŸ
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
    Draws an oriented bounding box around a vehicle on the given image and optionally labels it with its type.
    Args:
        image (np.ndarray): The image on which to draw the bounding box.
        Xvehicle (float or int): The X coordinate of the vehicle's center (used for label placement).
        Yvehicle (float or int): The Y coordinate of the vehicle's center (used for label placement).
        pt1 (tuple): The first corner point (row, col) of the bounding box.
        pt2 (tuple): The second corner point (row, col) of the bounding box.
        pt3 (tuple): The third corner point (row, col) of the bounding box.
        pt4 (tuple): The fourth corner point (row, col) of the bounding box.
        veh_type (str or None): The vehicle type code (e.g., '001' for Car, '002' for Truck, etc.).
        color (tuple, optional): The color of the bounding box lines in BGR format. Default is (0, 255, 0).
        thickness (int, optional): The thickness of the bounding box lines. Default is 1.
    Returns:
        np.ndarray: The image with the drawn bounding box (and optionally the label).
    Notes:
        - The function prints the area (in pixels) of the oriented bounding box.
        - The label drawing code is currently commented out.
        - Vehicle type codes are mapped to human-readable labels.
    """
    
    # Konvertiere die Punktkoordinaten in Integer-Tupel fÃ¼r cv2.line
    p1 = (int(pt1[1]), int(pt1[0]))
    p2 = (int(pt2[1]), int(pt2[0]))
    p3 = (int(pt3[1]), int(pt3[0]))
    p4 = (int(pt4[1]), int(pt4[0]))

    color = (0, 0, 0)
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

    # if label:
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     font_scale = 0.5
    #     font_thickness = 1
    #     text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    #     text_x = int(Xvehicle + 15)
    #     text_y = int(Yvehicle - 15)
    #     text_bg_color = (0, 0, 0)  # Schwarz
    #     text_color = (255, 255, 255)  # WeiÃŸ
    #     padding = 2

    #     #Zeichne den Hintergrund des Textes
    #     cv2.rectangle(image,
    #                   (text_x - padding, text_y - text_size[1] - padding),
    #                   (text_x + text_size[0] + padding, text_y + padding),
    #                   text_bg_color, cv2.FILLED)

    #     #Zeichne den Text
    #     cv2.putText(image, label, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return image


def get_bounding_box_in_px(img, label, oriented, ret_pts):
    """
    Calculates the bounding box of a vehicle in pixel coordinates from a label string and image.
    Args:
        img (np.ndarray): The image array on which the bounding box may be drawn.
        label (str): A string containing vehicle information, including position, orientation, and corner coordinates.
        oriented (bool): If True, returns or draws an oriented bounding box; if False, returns or draws an axis-aligned bounding box.
        ret_pts (bool): If True, returns the bounding box corner points; if False, draws the bounding box on the image.
    Returns:
        list or np.ndarray:
            - If ret_pts is True and oriented is True: Returns a list [veh_type, p1, p2, p3, p4] with vehicle type and oriented bounding box corner points.
            - If ret_pts is True and oriented is False: Returns a list [veh_type, p1, p2, p3, p4] with vehicle type and axis-aligned bounding box corner points.
            - If ret_pts is False and oriented is True: Draws the oriented bounding box on the image and returns the result.
            - If ret_pts is False and oriented is False: Draws the axis-aligned bounding box on the image and returns the result.
    Notes:
        - The function parses the label string to extract vehicle position, orientation, type, and corner coordinates.
        - Bounding box coordinates are calculated based on vehicle geometry and orientation.
        - Drawing functions `draw_oriented_vehicle_box` and `draw_axis_aligned_vehicle_bbox` must be defined elsewhere.
    """
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
    ltri = l[ktri]              # lÃ¤ngen Triangle

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
    Calculates the area (in pixels) of the bounding box that encloses a set of 2D points.
    Assumes points are in the format [y, x]. If the second column has a higher mean than the first,
    the columns are swapped to [x, y]. The function computes the minimum and maximum coordinates
    along both axes to determine the bounding box, then calculates its area.
    Parameters
    ----------
    points : list or numpy.ndarray
        A list or array of points, each represented as [y, x] or [x, y]. Must have shape (n, 2).
    Returns
    -------
    int
        The area of the bounding box in pixels. Returns 0 if points is None or empty.
    Raises
    ------
    ValueError
        If the input does not have shape (n, 2).
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
    Calculates the area of the oriented (minimum area) bounding box that encloses a set of 2D points.
    Args:
        points (array-like): An iterable of 2D points with shape (n, 2), where n >= 3.
    Returns:
        float: The area of the oriented bounding box. Returns 0.0 if fewer than 3 points are provided.
    Raises:
        ValueError: If the input points do not have shape (n, 2).
    Notes:
        - The function first computes the convex hull of the input points.
        - It then finds the minimum-area rectangle that encloses the convex hull using OpenCV's `minAreaRect`.
        - The area is calculated as width * height of the rectangle.
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


def select_all_labels_in_img(target, labels):
    """
    Filters and returns all labels from the given list that start with the specified target prefix.

    Args:
        target (str): The prefix to match at the start of each label.
        labels (list of str): The list of label strings to filter.

    Returns:
        list of str: A list containing all labels that start with the target prefix.
    """
    # Filtert alle Labels, die mit den Anfangsziffern von 'target' Ã¼bereinstimmen
    filtered_labels = [label for label in labels if label.startswith(target)]
    return filtered_labels

 
def transform_labels_to_yolo_format(labels, width, height):
    """
    Converts a list of label strings into YOLO format.
    Each label string is expected to contain at least 13 space-separated values:
    - The first value is the class ID (integer).
    - The second and third values are the x and y center coordinates (floats).
    - The next eight values are the coordinates of four points (x1, y1, x2, y2, x3, y3, x4, y4) as integers.
    The function parses these values, converts them to the required types, and uses the
    `convert_to_yolo_classic` function to transform them into YOLO format, normalizing
    coordinates based on the provided image width and height.
    Invalid labels or parsing errors are reported via printed messages.
    Args:
        labels (list of str): List of label strings to be converted.
        width (int): Width of the image for normalization.
        height (int): Height of the image for normalization.
    Returns:
        list: List of labels converted to YOLO format.
    """
    yolo_labels = []  # Liste fÃ¼r die konvertierten YOLO-Labels
    for label in labels:
        label_parts = label.split()  # Teilt das Label in Teile basierend auf Leerzeichen
        if len(label_parts) >= 13:  # Sicherstellen, dass genÃ¼gend Teile vorhanden sind
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
            print(f"UngÃ¼ltiges Label-Format: {label}")
    return yolo_labels

def convert_to_yolo_classic(x_center, y_center, x1, y1, x2, y2, x3, y3, x4, y4, class_id, img_width, img_height):
    """
    Converts bounding box coordinates and class ID to YOLO classic format.
    The function takes the center coordinates of the bounding box, the coordinates of its four corners,
    the class ID, and the image dimensions. It computes the minimum and maximum x and y values to
    determine the bounding box, normalizes the center coordinates and dimensions by the image size,
    and returns a formatted string in the YOLO classic annotation format.
    Args:
        x_center (float): X coordinate of the bounding box center.
        y_center (float): Y coordinate of the bounding box center.
        x1 (float): X coordinate of the first corner.
        y1 (float): Y coordinate of the first corner.
        x2 (float): X coordinate of the second corner.
        y2 (float): Y coordinate of the second corner.
        x3 (float): X coordinate of the third corner.
        y3 (float): Y coordinate of the third corner.
        x4 (float): X coordinate of the fourth corner.
        y4 (float): Y coordinate of the fourth corner.
        class_id (int): Class identifier for the object.
        img_width (int): Width of the image.
        img_height (int): Height of the image.
    Returns:
        str: A string in YOLO classic format: "<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>",
             where coordinates and dimensions are normalized to [0, 1] range.
    """
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
def show_one_picture_with_yolo_label(fold_nr,img_nr,ir, string_tag):
    def show_one_picture_with_yolo_label(fold_nr, img_nr, ir, string_tag):
        """
        Displays an image with YOLO label points overlaid.
        Args:
            fold_nr (int): The fold number to select the dataset partition.
            img_nr (int): The image number to select the specific image.
            ir (bool): If True, selects the IR image path (currently same as else).
            string_tag (str): Tag specifying the subfolder (e.g., 'train', 'val', 'test').
        Reads the image and its corresponding YOLO label file, overlays the labeled points
        on the image, and displays it in a window.
        Note:
            - The function expects images and labels to be stored in a specific folder structure.
            - Requires OpenCV (`cv2`) and helper functions `read_file` and 
              `add_labeled_normalized_coordinates_as_points`.
        """
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
    Draws labeled points on an image using normalized coordinates.
    Args:
        image_path (str): Path to the image file.
        labels_string (str): String containing label information, where normalized coordinates (x, y) start from the second element.
        point_color (tuple, optional): BGR color tuple for the points. Defaults to (0, 0, 255) (red).
        point_radius (int, optional): Radius of the drawn points. Defaults to 5.
        point_thickness (int, optional): Thickness of the points. Defaults to -1 (filled circle).
    Returns:
        numpy.ndarray or None: The image with drawn points, or None if the image could not be loaded.
    Raises:
        Exception: If an error occurs during processing.
    """
   
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Fehler: Bild nicht gefunden unter {image_path}")
            return

        height, width, _ = img.shape

        # Teile die Label-Zeichenkette auf und Ã¼berspringe das erste Element
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




def create_yaml(path, fold_nr, namestring, palma, fold_dest_path, test_dataset_bool, number_of_channels):
    """
    Creates a YAML configuration file for dataset splits and class information.
    Args:
        path (str): Directory path where the YAML file will be saved.
        fold_nr (int): Fold number for cross-validation.
        namestring (str): Additional string to append to the fold directory name.
        palma (bool): If True, use Palma cluster paths; otherwise, use local paths.
        fold_dest_path (str): Destination path for fold data (used in Palma mode).
        test_dataset_bool (bool): If True, include test dataset paths in the YAML file.
        number_of_channels (int): Number of image channels; adds 'channels' field if >3 or ==1.
    Side Effects:
        Writes a YAML file named 'data.yaml' to the specified path containing dataset split paths,
        number of classes, class names, and optionally the number of channels.
    Raises:
        IOError: If there is an error writing the YAML file.
    """
    file_path = f"{path}/data.yaml"
    number_channel_string = "channels: " + str(number_of_channels)

    if palma == True:
        train_image_path = f"/scratch/tmp/t_liet02/{fold_dest_path}/fold{fold_nr}{namestring}/train/images"
        val_image_path = f"/scratch/tmp/t_liet02/{fold_dest_path}/fold{fold_nr}{namestring}/val/images"
        test_image_path = f"/scratch/tmp/t_liet02/{fold_dest_path}/fold{fold_nr}{namestring}/test/images"
        
        if test_dataset_bool:
            file_string = "train: "+train_image_path +'\n'+ "val: " +val_image_path +'\n'+"test: "+test_image_path +'\n' +  "nc: 9"  +'\n'+"names: ['Car', 'Truck', 'Ship', 'Tractor', 'Camping Car', 'van', 'vehicle', 'pick-up', 'plane']"
        else:
            file_string = "train: "+train_image_path +'\n'+ "val: " +val_image_path +'\n'+  "nc: 9"  +'\n'+"names: ['Car', 'Truck', 'Ship', 'Tractor', 'Camping Car', 'van', 'vehicle', 'pick-up', 'plane']"
    else:
        if test_dataset_bool:
            file_string = "train: ./train/images " +'\n'+ "val: ./val/images" +'\n' + "test: ./test/images" + '\n'+  "nc: 9"  +'\n'+"names: ['Car', 'Truck', 'Ship', 'Tractor', 'Camping Car', 'van', 'vehicle', 'pick-up', 'plane']"
        else:
            file_string = "train: ./train/images " +'\n'+ "val: ./val/images" + '\n'+  "nc: 9"  +'\n'+"names: ['Car', 'Truck', 'Ship', 'Tractor', 'Camping Car', 'van', 'vehicle', 'pick-up', 'plane']"
    
    if number_of_channels > 3 or number_of_channels == 1:
        file_string += '\n'+ number_channel_string

    try:
        with open(file_path, 'w') as file:
            file.write(file_string)
    except IOError as e:
        print(f"Fehler beim Schreiben der YAML-Datei: {e}")

def read_file(path):
    """
    Reads a text file and returns a list of its lines with leading and trailing whitespace removed.

    Args:
        path (str): The path to the text file to be read.

    Returns:
        list of str: A list containing each line of the file, stripped of leading and trailing whitespace.
    """
    with open(path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file]  # Entfernt \n aus allen Zeilen
    return lines



def create_fold_cross_validation(
        fold_nr,
        path_all_images,
        path_labels,
        ir,
        fold10bool,
        bool_create_yaml,
        limiter,
        oriented,
        merge_ir_bool,
        namestring,
        palma_bool,
        fold_dest_path,
        perm_object
    ):
    """
    Creates cross-validation folds for image and label datasets, organizing images and labels into train, validation, and test splits.
    Args:
        fold_nr (int): The current fold number to be used as validation.
        path_all_images (str): Path to the directory containing all images.
        path_labels (str): Path to the file containing all labels.
        ir (bool): Whether to include infrared images.
        fold10bool (bool): If True, use 10-fold cross-validation logic.
        bool_create_yaml (bool): If True, create a YAML configuration file for the fold.
        limiter (int or None): Limits the number of images processed per split; if None, all images are processed.
        oriented (bool): Whether to use oriented bounding boxes for labels.
        merge_ir_bool (bool): If True, merge IR images with color images.
        namestring (str): String to append to fold names and paths.
        palma_bool (bool): If True, use PALMA cluster paths; otherwise, use local paths.
        fold_dest_path (str): Destination path for the fold output.
        perm_object (dict or None): Dictionary specifying which image channels to include.
    Returns:
        int: Returns 0 upon successful creation of the fold.
    Side Effects:
        - Copies images and creates label files for train, validation, and test splits.
        - Optionally creates a YAML configuration file for the fold.
        - Prints progress and status messages to the console.
        - Creates necessary folder structures for the fold output.
    Notes:
        - Assumes existence of helper functions: create_folder_structure, create_yaml, read_file, select_all_labels_in_img, copy_image, create_label_file.
        - The function processes images and labels according to the specified fold and configuration.
        - The train split consists of all folds except the current validation fold.
        - The test split is always taken from fold 5.
    """
    def create_image_and_label(lines,  string_tag, current_fold_nr):
        """
        Processes a list of image identifiers, copies corresponding images, creates label files, and prints progress information.
        Args:
            lines (list): List of image identifiers (strings) to process.
            string_tag (str): Tag indicating the dataset split ('train', 'val', or 'test').
            current_fold_nr (int or None): Current fold number for cross-validation, or None if not applicable.
        Side Effects:
            - Copies images to designated directories.
            - Creates label files for each image.
            - Prints progress information to the console.
            - Applies a limiter to restrict the number of processed images for each split.
        Notes:
            - Uses global variables and functions: path_all_images, labels, select_all_labels_in_img, copy_image, 
              paths_object, merge_ir_bool, perm_object, create_label_file, ir, oriented, fold_nr, limiter, np.
            - The limiter restricts the number of processed images for 'train', 'val', and 'test' splits differently.
        """
        counter = 0
        for line in lines:
            counter += 1
            target = line
          
            image_path = f"{path_all_images}/{target}_co.png"         
            image_path_ir = f"{path_all_images}/{target}_ir.png"

            filtered_labels = select_all_labels_in_img(target, labels)
            copy_image(image_path, paths_object['path_'+string_tag+'_images'], merge_ir_bool, image_path, image_path_ir, perm_object)
            create_label_file(target, filtered_labels, paths_object['path_'+string_tag+'_labels'], image_path, ir, oriented, string_tag)
            if current_fold_nr is not None:
                print("Fold Nr:"+str(fold_nr)+" /  "+ string_tag+" image: "+str(counter) + "/" + str(len(lines))+ " from train fold: " + str(current_fold_nr))
            else:
                print("Fold Nr:"+str(fold_nr)+" /  "+ string_tag+" image: "+str(counter) + "/" + str(len(lines)))
            
            if limiter != None:
                if counter == limiter and string_tag == 'train':
                    print("Limiter! BREAK "+ string_tag+" at " + str(limiter))
                    break
                elif counter == np.ceil(limiter/10).astype(int) and string_tag == 'val':
                    print("Limiter! BREAK "+ string_tag+" at " + str(limiter))
                    break
                elif counter == np.ceil(limiter/10).astype(int) and string_tag == 'test':
                    print("Limiter! BREAK "+ string_tag+" at " + str(limiter))
                    break
            

    if palma_bool == True:
        fold_val_images_path = rf'../../../scratch/tmp/t_liet02/folds/txts/fold{fold_nr}.txt'
        fold_test_images_path = rf'../../../scratch/tmp/t_liet02/folds/txts/fold5.txt'
        yaml_path = rf'../../../scratch/tmp/t_liet02/{fold_dest_path}/fold{fold_nr}{namestring}'
    elif fold10bool == True and palma_bool == False:
        fold_val_images_path = rf'Code\data\folds\own_folds\fold{fold_nr}.txt'
        fold_test_images_path = rf'Code\data\folds\own_folds\fold5.txt'
        yaml_path = rf'Code\data\folds\{fold_dest_path}\fold{fold_nr}{namestring}'
   
    number_of_channels = 3	

    paths_object = create_folder_structure(fold_nr, namestring, palma_bool, fold_dest_path, True)
    if perm_object is not None:
        number_of_channels = sum(1 for v in perm_object.values() if v)
	    
    
    if bool_create_yaml:
        create_yaml(yaml_path, fold_nr, namestring, palma_bool, fold_dest_path, True, number_of_channels)
        print("Yaml successfull created.")

    all_fold_nr = list(range(0,5))
    all_fold_nr.remove(fold_nr)

    all_fold_nr_without_current_fold = all_fold_nr
    all_fold_nr = list(range(0,5))

    labels = read_file(path_labels)


    for i in all_fold_nr_without_current_fold:


        if palma_bool == True:
            fold_train_images_path = rf'../../../scratch/tmp/t_liet02/folds/txts/fold{i}.txt'
        elif palma_bool == False:
            fold_train_images_path = rf'Code\data\folds\own_folds\fold{i}.txt'
       
        lines_fold_train = read_file(fold_train_images_path)
        create_image_and_label(lines_fold_train, "train", i)

    lines_fold_val = read_file(fold_val_images_path)
    lines_fold_test  = read_file(fold_test_images_path)

    create_image_and_label(lines_fold_val, "val", None)
    create_image_and_label(lines_fold_test, "test", None)
   
    print("Fold " + str(fold_nr) + " / " + namestring + " in ' "+fold_dest_path+ " successfull created.")
    return 0

def print_divideline():
    """
    Prints a long horizontal dividing line to the console for visual separation of output sections.
    """
    print("________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________")

if __name__ == "__main__":
    main()