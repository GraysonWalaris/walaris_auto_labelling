import numpy as np
import glob
import os
from tqdm import tqdm

from argparse import ArgumentParser
import torch.distributed as dist

import numpy as np
from data_processing import view, labels
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import json
import random

WALARIS_CLASS_LABELS_NUM2NAME = labels.WALARIS_CLASS_LABELS_NUM2NAME
WALARIS_CLASS_LABELS_NAME2NUM = labels.WALARIS_CLASS_LABELS_NAME2NUM

###############################################################################
                     # COCO Class Dictionaries #
###############################################################################
MMDET_COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic_light",
    "fire_hydrant",
    "stop_sign",
    "parking_meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports_ball",
    "kite",
    "baseball_bat",
    "baseball_glove",
    "skateboard",
    "surfboard",
    "tennis_racket",
    "bottle",
    "wine_glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot_dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted_plant",
    "bed",
    "dining_table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell_phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy_bear",
    "hair_drier",
    "toothbrush",
]

MMDET_COCO_CLASSES_DICT_NAME2NUM = {}
MMDET_COCO_CLASSES_DICT_NUM2NAME = {}
for num, class_type in enumerate(MMDET_COCO_CLASSES):
    MMDET_COCO_CLASSES_DICT_NAME2NUM[class_type] = num
    MMDET_COCO_CLASSES_DICT_NUM2NAME[num] = class_type

COCO_CATEGORIES = [
    {"supercategory": "person","id": 1,"name": "person"},
    {"supercategory": "vehicle","id": 2,"name": "bicycle"},
    {"supercategory": "vehicle","id": 3,"name": "car"},
    {"supercategory": "vehicle","id": 4,"name": "motorcycle"},
    {"supercategory": "vehicle","id": 5,"name": "airplane"},
    {"supercategory": "vehicle","id": 6,"name": "bus"},
    {"supercategory": "vehicle","id": 7,"name": "train"},
    {"supercategory": "vehicle","id": 8,"name": "truck"},
    {"supercategory": "vehicle","id": 9,"name": "boat"},
    {"supercategory": "outdoor","id": 10,"name": "traffic light"},
    {"supercategory": "outdoor","id": 11,"name": "fire hydrant"},
    {"supercategory": "outdoor","id": 13,"name": "stop sign"},
    {"supercategory": "outdoor","id": 14,"name": "parking meter"},
    {"supercategory": "outdoor","id": 15,"name": "bench"},
    {"supercategory": "animal","id": 16,"name": "bird"},
    {"supercategory": "animal","id": 17,"name": "cat"},
    {"supercategory": "animal","id": 18,"name": "dog"},
    {"supercategory": "animal","id": 19,"name": "horse"},
    {"supercategory": "animal","id": 20,"name": "sheep"},
    {"supercategory": "animal","id": 21,"name": "cow"},
    {"supercategory": "animal","id": 22,"name": "elephant"},
    {"supercategory": "animal","id": 23,"name": "bear"},
    {"supercategory": "animal","id": 24,"name": "zebra"},
    {"supercategory": "animal","id": 25,"name": "giraffe"},
    {"supercategory": "accessory","id": 27,"name": "backpack"},
    {"supercategory": "accessory","id": 28,"name": "umbrella"},
    {"supercategory": "accessory","id": 31,"name": "handbag"},
    {"supercategory": "accessory","id": 32,"name": "tie"},
    {"supercategory": "accessory","id": 33,"name": "suitcase"},
    {"supercategory": "sports","id": 34,"name": "frisbee"},
    {"supercategory": "sports","id": 35,"name": "skis"},
    {"supercategory": "sports","id": 36,"name": "snowboard"},
    {"supercategory": "sports","id": 37,"name": "sports ball"},
    {"supercategory": "sports","id": 38,"name": "kite"},
    {"supercategory": "sports","id": 39,"name": "baseball bat"},
    {"supercategory": "sports","id": 40,"name": "baseball glove"},
    {"supercategory": "sports","id": 41,"name": "skateboard"},
    {"supercategory": "sports","id": 42,"name": "surfboard"},
    {"supercategory": "sports","id": 43,"name": "tennis racket"},
    {"supercategory": "kitchen","id": 44,"name": "bottle"},
    {"supercategory": "kitchen","id": 46,"name": "wine glass"},
    {"supercategory": "kitchen","id": 47,"name": "cup"},
    {"supercategory": "kitchen","id": 48,"name": "fork"},
    {"supercategory": "kitchen","id": 49,"name": "knife"},
    {"supercategory": "kitchen","id": 50,"name": "spoon"},
    {"supercategory": "kitchen","id": 51,"name": "bowl"},
    {"supercategory": "food","id": 52,"name": "banana"},
    {"supercategory": "food","id": 53,"name": "apple"},
    {"supercategory": "food","id": 54,"name": "sandwich"},
    {"supercategory": "food","id": 55,"name": "orange"},
    {"supercategory": "food","id": 56,"name": "broccoli"},
    {"supercategory": "food","id": 57,"name": "carrot"},
    {"supercategory": "food","id": 58,"name": "hot dog"},
    {"supercategory": "food","id": 59,"name": "pizza"},
    {"supercategory": "food","id": 60,"name": "donut"},
    {"supercategory": "food","id": 61,"name": "cake"},
    {"supercategory": "furniture","id": 62,"name": "chair"},
    {"supercategory": "furniture","id": 63,"name": "couch"},
    {"supercategory": "furniture","id": 64,"name": "potted plant"},
    {"supercategory": "furniture","id": 65,"name": "bed"},
    {"supercategory": "furniture","id": 67,"name": "dining table"},
    {"supercategory": "furniture","id": 70,"name": "toilet"},
    {"supercategory": "electronic","id": 72,"name": "tv"},
    {"supercategory": "electronic","id": 73,"name": "laptop"},
    {"supercategory": "electronic","id": 74,"name": "mouse"},
    {"supercategory": "electronic","id": 75,"name": "remote"},
    {"supercategory": "electronic","id": 76,"name": "keyboard"},
    {"supercategory": "electronic","id": 77,"name": "cell phone"},
    {"supercategory": "appliance","id": 78,"name": "microwave"},
    {"supercategory": "appliance","id": 79,"name": "oven"},
    {"supercategory": "appliance","id": 80,"name": "toaster"},
    {"supercategory": "appliance","id": 81,"name": "sink"},
    {"supercategory": "appliance","id": 82,"name": "refrigerator"},
    {"supercategory": "indoor","id": 84,"name": "book"},
    {"supercategory": "indoor","id": 85,"name": "clock"},
    {"supercategory": "indoor","id": 86,"name": "vase"},
    {"supercategory": "indoor","id": 87,"name": "scissors"},
    {"supercategory": "indoor","id": 88,"name": "teddy bear"},
    {"supercategory": "indoor","id": 89,"name": "hair drier"},
    {"supercategory": "indoor","id": 90,"name": "toothbrush"}
]

COCO_CLASSES_DICT_NAME2NUM = {}
COCO_CLASSES_DICT_NUM2NAME = {}
for category in COCO_CATEGORIES:
    COCO_CLASSES_DICT_NAME2NUM[category['name']] = category['id']
    COCO_CLASSES_DICT_NUM2NAME[category['id']] = category['name']

MAPPING_WALARIS_TO_COCO = {
    1: 5,   # uav (1) -> airplance (5)
    2: 5,   # airplane (2) -> airplane (5)
    3: 2,   # bicycle (3) -> bicycle (2)
    4: 16,  # bird (4) -> bird (16)
    5: 9,   # boat (5) -> boat (9)
    6: 6,   # bus (6) -> bus (6)
    7: 3,   # car (7) -> car (3)
    8: 17,  # cat (8) -> cat (17)
    9: 21,  # cow (9) -> cow (21)
    10: 18, # dog (10) -> dog (18)
    11: 19, # horse (11) -> horse (19)
    12: 4,  # motorcycle (12) -> motorcycle (4)
    13: 1,  # person (13) -> person (1)
    14: 10, # traffic_light (14) -> traffic light (10)
    15: 7,  # train (15) -> train (7)
    16: 8,  # truck (16) -> truck (8)
    17: 16, # ufo (17) -> bird (16)
    18: 5   # helicopter (18) -> airplane (5)
}

###############################################################################
                     # YOLOv8 Class Dictionaries #
###############################################################################

def get_results_over_thr(result,
                         score_thr=0.3,
                        ):
    """Takes in mmdet results object and a score threshold and returns the 
    bboxes, labels, and masks that are above the threshold.
    
    Parameters:
        results (mmdet result object): results from mmdet.apis.inference_detector
        score_thr (float): threshold that the scores must be above to be returned

    Returns:
        bboxes (list): bboxes above the threshold
        labels (list): labels above the threshold
        segms (list): segms above the threshold
    """

    bboxes = result.pred_instances.bboxes
    labels = result.pred_instances.labels
    scores = result.pred_instances.scores

    assert len(bboxes) == len(labels) == len(scores), "Error: Different number \
        of bboxes, labels, and scores!"

    for idx, score in enumerate(scores):
        if score < score_thr:
            bboxes = bboxes[:idx]
            labels = labels[:idx]
            scores = scores[:idx]
            break

    return bboxes, labels, scores

def get_path_of_images():
    """Gets the path of every single image in the dataset. Returns a list of
    file paths.
    
    Parameters:

    Returns:
        image_paths (list): list of every image path in the dataset
    """
    base_image_path = os.path.join(os.environ.get('WALARIS_MAIN_DATA_PATH'),
                                   'Images')
    
    image_path_list = []
    for class_folder in glob.glob(base_image_path+'/*'):
        print(f'Getting {class_folder.split("/")[-1]} images..')
        for img_folder in tqdm(glob.glob(class_folder+'/*')):
            for img in glob.glob(img_folder+'/*.png'):
                # reduce the file path to get just the extension beyond 
                # Tarsier_Main_Datset/Images/ to save RAM
                reduced_file_path = img.split('/')
                reduced_file_path = ('/').join(reduced_file_path[-3:])
                image_path_list.append(reduced_file_path)

    return image_path_list

def log_experiment_info(args, last_image_inferenced, start_time):
    exp_info = {
        'model_config': args.config,
        'model_checkpoint': args.checkpoint,
        'last_image_inferenced': str(last_image_inferenced)
    }

    labels.save_dict_to_json(f'results/{str(start_time)}/log.json', 
                             exp_info,
                             delete=True)
    
def convert_format_walaris_to_coco(json_file,
                                   new_json_file):
    # Read the JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Create a dictionary for the custom dataset
    custom_dataset = {}

    # Set the dataset information
    custom_dataset["info"] = {
        "description": "My COCO Dataset",
        "version": "1.0",
        "year": 2023,
        "contributor": "Walaris",
        "date_created": "2023-06-20"
    }

    # Set the license information
    custom_dataset["licenses"] = [
        {
            "id": 1,
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT"
        }
    ]

    # Process the images
    custom_dataset["images"] = []
    for dict_img in data["images"]:
        temp = {
            "license": 1,
            "file_name": dict_img["file_name"],
            "height": dict_img["height"],
            "width": dict_img["width"],
            "id": dict_img["id"]
        }
        custom_dataset["images"].append(temp)

    # Set the categories from COCO dataset
    custom_dataset["categories"] = [
        {"supercategory": "person","id": 1,"name": "person"},
        {"supercategory": "vehicle","id": 2,"name": "bicycle"},
        {"supercategory": "vehicle","id": 3,"name": "car"},
        {"supercategory": "vehicle","id": 4,"name": "motorcycle"},
        {"supercategory": "vehicle","id": 5,"name": "airplane"},
        {"supercategory": "vehicle","id": 6,"name": "bus"},
        {"supercategory": "vehicle","id": 7,"name": "train"},
        {"supercategory": "vehicle","id": 8,"name": "truck"},
        {"supercategory": "vehicle","id": 9,"name": "boat"},
        {"supercategory": "outdoor","id": 10,"name": "traffic light"},
        {"supercategory": "outdoor","id": 11,"name": "fire hydrant"},
        {"supercategory": "outdoor","id": 13,"name": "stop sign"},
        {"supercategory": "outdoor","id": 14,"name": "parking meter"},
        {"supercategory": "outdoor","id": 15,"name": "bench"},
        {"supercategory": "animal","id": 16,"name": "bird"},
        {"supercategory": "animal","id": 17,"name": "cat"},
        {"supercategory": "animal","id": 18,"name": "dog"},
        {"supercategory": "animal","id": 19,"name": "horse"},
        {"supercategory": "animal","id": 20,"name": "sheep"},
        {"supercategory": "animal","id": 21,"name": "cow"},
        {"supercategory": "animal","id": 22,"name": "elephant"},
        {"supercategory": "animal","id": 23,"name": "bear"},
        {"supercategory": "animal","id": 24,"name": "zebra"},
        {"supercategory": "animal","id": 25,"name": "giraffe"},
        {"supercategory": "accessory","id": 27,"name": "backpack"},
        {"supercategory": "accessory","id": 28,"name": "umbrella"},
        {"supercategory": "accessory","id": 31,"name": "handbag"},
        {"supercategory": "accessory","id": 32,"name": "tie"},
        {"supercategory": "accessory","id": 33,"name": "suitcase"},
        {"supercategory": "sports","id": 34,"name": "frisbee"},
        {"supercategory": "sports","id": 35,"name": "skis"},
        {"supercategory": "sports","id": 36,"name": "snowboard"},
        {"supercategory": "sports","id": 37,"name": "sports ball"},
        {"supercategory": "sports","id": 38,"name": "kite"},
        {"supercategory": "sports","id": 39,"name": "baseball bat"},
        {"supercategory": "sports","id": 40,"name": "baseball glove"},
        {"supercategory": "sports","id": 41,"name": "skateboard"},
        {"supercategory": "sports","id": 42,"name": "surfboard"},
        {"supercategory": "sports","id": 43,"name": "tennis racket"},
        {"supercategory": "kitchen","id": 44,"name": "bottle"},
        {"supercategory": "kitchen","id": 46,"name": "wine glass"},
        {"supercategory": "kitchen","id": 47,"name": "cup"},
        {"supercategory": "kitchen","id": 48,"name": "fork"},
        {"supercategory": "kitchen","id": 49,"name": "knife"},
        {"supercategory": "kitchen","id": 50,"name": "spoon"},
        {"supercategory": "kitchen","id": 51,"name": "bowl"},
        {"supercategory": "food","id": 52,"name": "banana"},
        {"supercategory": "food","id": 53,"name": "apple"},
        {"supercategory": "food","id": 54,"name": "sandwich"},
        {"supercategory": "food","id": 55,"name": "orange"},
        {"supercategory": "food","id": 56,"name": "broccoli"},
        {"supercategory": "food","id": 57,"name": "carrot"},
        {"supercategory": "food","id": 58,"name": "hot dog"},
        {"supercategory": "food","id": 59,"name": "pizza"},
        {"supercategory": "food","id": 60,"name": "donut"},
        {"supercategory": "food","id": 61,"name": "cake"},
        {"supercategory": "furniture","id": 62,"name": "chair"},
        {"supercategory": "furniture","id": 63,"name": "couch"},
        {"supercategory": "furniture","id": 64,"name": "potted plant"},
        {"supercategory": "furniture","id": 65,"name": "bed"},
        {"supercategory": "furniture","id": 67,"name": "dining table"},
        {"supercategory": "furniture","id": 70,"name": "toilet"},
        {"supercategory": "electronic","id": 72,"name": "tv"},
        {"supercategory": "electronic","id": 73,"name": "laptop"},
        {"supercategory": "electronic","id": 74,"name": "mouse"},
        {"supercategory": "electronic","id": 75,"name": "remote"},
        {"supercategory": "electronic","id": 76,"name": "keyboard"},
        {"supercategory": "electronic","id": 77,"name": "cell phone"},
        {"supercategory": "appliance","id": 78,"name": "microwave"},
        {"supercategory": "appliance","id": 79,"name": "oven"},
        {"supercategory": "appliance","id": 80,"name": "toaster"},
        {"supercategory": "appliance","id": 81,"name": "sink"},
        {"supercategory": "appliance","id": 82,"name": "refrigerator"},
        {"supercategory": "indoor","id": 84,"name": "book"},
        {"supercategory": "indoor","id": 85,"name": "clock"},
        {"supercategory": "indoor","id": 86,"name": "vase"},
        {"supercategory": "indoor","id": 87,"name": "scissors"},
        {"supercategory": "indoor","id": 88,"name": "teddy bear"},
        {"supercategory": "indoor","id": 89,"name": "hair drier"},
        {"supercategory": "indoor","id": 90,"name": "toothbrush"}
    ]
    # ------------ CUSTOM DATASET -----------                       --- COCO DATASET ---
    # [{'supercategory': 'none', 'name': 'uav', 'id': 1},           -> 5 (airplane)
    # {'supercategory': 'none', 'name': 'airplane', 'id': 2},       -> 5 (airplane)
    # {'supercategory': 'none', 'name': 'bicycle', 'id': 3},        -> 2 (bicycle)
    # {'supercategory': 'none', 'name': 'bird', 'id': 4},           -> 16 (bird)
    # {'supercategory': 'none', 'name': 'boat', 'id': 5},           -> 9 (boat)
    # {'supercategory': 'none', 'name': 'bus', 'id': 6},            -> 6 (bus)
    # {'supercategory': 'none', 'name': 'car', 'id': 7},            -> 3 (car)
    # {'supercategory': 'none', 'name': 'cat', 'id': 8},            -> 17 (cat)
    # {'supercategory': 'none', 'name': 'cow', 'id': 9},            -> 21 (cow)
    # {'supercategory': 'none', 'name': 'dog', 'id': 10},           -> 18 (dog)
    # {'supercategory': 'none', 'name': 'horse', 'id': 11},         -> 19 (horse)
    # {'supercategory': 'none', 'name': 'motorcycle', 'id': 12},    -> 4 (motorcycle)
    # {'supercategory': 'none', 'name': 'person', 'id': 13},        -> 1 (person)
    # {'supercategory': 'none', 'name': 'traffic_light', 'id': 14}, -> 10 (traffic light)
    # {'supercategory': 'none', 'name': 'train', 'id': 15},         -> 7 (train)
    # {'supercategory': 'none', 'name': 'truck', 'id': 16},         -> 8 (truck)
    # {'supercategory': 'none', 'name': 'ufo', 'id': 17},           -> 16 (bird)
    # {'supercategory': 'none', 'name': 'helicopter', 'id': 18}]    -> 5 (airplane)

    # Mapping dictionary from custom dataset IDs to COCO dataset IDs
    mapping_custom_to_COCO = {
        1: 5,   # uav (1) -> airplance (5)
        2: 5,   # airplane (2) -> airplane (5)
        3: 2,   # bicycle (3) -> bicycle (2)
        4: 16,  # bird (4) -> bird (16)
        5: 9,   # boat (5) -> boat (9)
        6: 6,   # bus (6) -> bus (6)
        7: 3,   # car (7) -> car (3)
        8: 17,  # cat (8) -> cat (17)
        9: 21,  # cow (9) -> cow (21)
        10: 18, # dog (10) -> dog (18)
        11: 19, # horse (11) -> horse (19)
        12: 4,  # motorcycle (12) -> motorcycle (4)
        13: 1,  # person (13) -> person (1)
        14: 10, # traffic_light (14) -> traffic light (10)
        15: 7,  # train (15) -> train (7)
        16: 8,  # truck (16) -> truck (8)
        17: 16, # ufo (17) -> bird (16)
        18: 5   # helicopter (18) -> airplane (5)
    }

    # Process the annotations
    custom_dataset["annotations"] = []
    for dict_ann in data["annotations"]:
        temp = {
            "segmentation": None,
            "area": dict_ann["area"],
            "iscrowd": dict_ann["iscrowd"],
            "image_id": dict_ann["image_id"],
            "bbox": dict_ann["bbox"],
            "category_id": mapping_custom_to_COCO[dict_ann["category_id"]],
            "id": dict_ann["id"]
        }
        custom_dataset["annotations"].append(temp)

    # Save the dictionary as a JSON file
    with open(new_json_file, 'w') as file:
        json.dump(custom_dataset, file)

def clean_json_dataset(original_json_file, 
                       new_json_file,
                       label_format,
                       classes_to_remove: set = None):
    """Takes a json file, reads it, and cleans it to remove all of the images
    that have not been downloaded on the machine. If the classes_to_remove
    contains a set, it will remove all annotations that belong to any class in
    the set.
    """

    with open(original_json_file, 'r') as file:
        data = json.load(file)

    images = data['images']
    categories = data['categories']
    annotations = data['annotations']

    # get set of all img paths to test
    img_paths = set()
    # create a dictionary that maps id to image path
    id2path = {}
    count = 0
    for img_info in images:
        img_path = img_info['file_name']
        img_id = img_info['id']
        id2path[img_id] = img_path
        if img_path in img_paths:
            count += 1
            print(f'Duplicate: {count}')
        else:
            img_paths.add(img_path)

    total = len(img_paths)
    base_path = os.path.join(os.environ.get('WALARIS_MAIN_DATA_PATH'),
                                    'Images')

    valid_img_paths = set()
    new_images = []
    for image in images:
        img_path_ext = image['file_name']
        full_path = os.path.join(base_path,
                                 img_path_ext)
        if os.path.exists(full_path):
            valid_img_paths.add(img_path_ext)
            new_images.append(image)
    
    new_annotations = []
    for annotation in annotations:
        annotation_img_id = annotation['image_id']
        img_path = id2path[annotation_img_id]
        if img_path in valid_img_paths:
            new_annotations.append(annotation)

    new_images = sorted(new_images, key=lambda x: x['id']) 
    new_annotations = sorted(new_annotations, key=lambda x: x['image_id'])

    data['images'] = new_images
    data['annotations'] = new_annotations

    with open(new_json_file, 'w') as file:
        json.dump(data, file)

    if classes_to_remove is not None:
        remove_coco_format_annotations_by_class(new_json_file,
                                                new_json_file,
                                                classes_to_remove,
                                                label_format=label_format)

def get_img_path_info(images):
    """ Gets a set of all of the image paths in a COCO['images'] list. Also
    returns a dictionary linking the image id and image paths of each image.

    Args:
        images (list): list of image dictionaries in the COCO format

    Returns:
        img_paths (set): set of all image paths in the images list
        id2img_path (dict): dictionary linking image ids to image paths
    
    """
    # get set of all img paths to test
    img_paths = set()
    # get dictionary mapping img ids to img path extensions
    id2img_path = dict()
    count = 0
    for img_info in images:
        img_id = img_info['id']
        img_path = img_info['file_name']
        # count duplicates
        if img_path in img_paths:
            count += 1
            
        else:
            id2img_path[img_id] = img_path
            img_paths.add(img_path)
    
    return img_paths, id2img_path 

def result_mmdet_2_coco_format(result, 
                               score_thr, 
                               img_id,
                               label_convention):
    """Takes in a mmdet result and threshold, returns a list of dictionaries in
    the coco results format. ie.

    [{
        "image_id": int, 
        "category_id": int, 
        "bbox": [x,y,width,height], 
        "score": float,
    }]

    Parameters
        result (tuple): (bboxes, class_labels, scores) * all of these are lists
        score_thr (float): predictions with a confidence score lower than this
         will be filtered out
        img_id (str): the image id of the label
        label_convention (str): specify whether you want the labels to follow
            the walaris standard convention or COCO standard convention

    Returns
        coco_result (List(Dict)): results in the coco results format.
    """
    bboxes, class_labels, scores = result

    assert len(bboxes) == len(class_labels) == len(scores), "Error: All result \
        lists must be the same length."
    
    # convert the class ids from mmdet_coco to true coco labels
    for idx, class_label in enumerate(class_labels):
        class_label_name = MMDET_COCO_CLASSES_DICT_NUM2NAME[class_label]
        if label_convention == 'coco':
            class_labels[idx] = COCO_CLASSES_DICT_NAME2NUM[class_label_name]
        elif label_convention == 'walaris':
            class_labels[idx] = WALARIS_CLASS_LABELS_NAME2NUM[class_label_name]

    coco_result = []
    for idx in range(len(bboxes)):
        # convert bbox: xyxy -> xywh
        x1, y1, x2, y2 = bboxes[idx]
        w = x2-x1
        h = y2-y1
        converted_bbox = [x1, y1, w, h]

        # create new result and add to coco_results
        result = {
            'image_id': img_id,
            'category_id': class_labels[idx],
            'bbox': converted_bbox,
            'score': scores[idx]
        }
        coco_result.append(result)

    return coco_result

def remove_unspecified_class_predictions(results,
                                         classes_to_predict,
                                         pred_format):
    """Removes any predictions of unspecified classes.
    
    Parameters
        results (tuple): (bboxes, class_labels, scores) * all are lists 
        classes_to_predicts (set): set of classes that you want to keep
         predictions for
        prediction_format (str): tells what format the predictions are in
         
    Returns
        results (tuple): (bboxes, class_labels, scores) * all are lists"""
    bboxes, class_labels, scores = results

    assert len(bboxes) == len(class_labels) == len(scores), "Error: All result \
        lists must be the same length."
    
    if pred_format == 'mmdet':
        num2name = MMDET_COCO_CLASSES_DICT_NUM2NAME
    if pred_format == 'coco':
        num2name = COCO_CLASSES_DICT_NUM2NAME

    # remove unwanted classes with in-place algorithm (2 ptr swap)
    left_ptr = 0
    right_ptr = 0
    while right_ptr < len(bboxes):
        class_name = num2name[class_labels[right_ptr]]
        if class_name in classes_to_predict:
            bboxes[left_ptr], bboxes[right_ptr] = bboxes[right_ptr], bboxes[left_ptr]
            class_labels[left_ptr], class_labels[right_ptr] = class_labels[right_ptr], class_labels[left_ptr]
            scores[left_ptr], scores[right_ptr] = scores[right_ptr], scores[left_ptr]

            # increment left ptr
            left_ptr += 1
        right_ptr += 1
    
    bboxes = bboxes[:left_ptr]
    class_labels = class_labels[:left_ptr]
    scores = scores[:left_ptr]

    return bboxes, class_labels, scores

def remove_coco_format_annotations_by_class(original_json_file: str,
                                            new_json_file: str,
                                            classes_to_remove: set,
                                            label_format):
    """Goes through a json file and removes all of the labels for classes in
    the classes_to_remove list and saves the refined labels to a new json file.
    
    Parameters
        original_json_file (str): Json file to read initial labels from
        new_json_file (str): save path for the refined labels
        classes_to_remove (set): set of classes to remove from the labels
        
    Returns
        None
    """

    if label_format == 'walaris':
        name2num = WALARIS_CLASS_LABELS_NAME2NUM
    elif label_format == 'coco':
        name2num = COCO_CLASSES_DICT_NAME2NUM
    elif label_format == 'mmdet':
        name2num = MMDET_COCO_CLASSES_DICT_NAME2NUM

    classes_to_remove_num = set()
    for class_label in classes_to_remove:
        classes_to_remove_num.add(name2num[class_label])


    with open(original_json_file, 'r') as file:
        original_info = json.load(file)

    # create a refined annotations list, keep track of all refined img_ids
    refined_annotations= []
    for label in original_info['annotations']:
        if label['category_id'] not in classes_to_remove_num:
            img_id = label['image_id']
            refined_annotations.append(label)

    # replace the original annotations and images with the refined ones
    original_info['annotations'] = refined_annotations

    # save refined dataset information to new file
    with open(new_json_file, 'w') as file:
        json.dump(original_info, file)

    return

def get_random_sample_from_json_file(original_json_file,
                                     new_json_file,
                                     sample_size,
                                     seed=None,
                                     include_unlabelled_images=False):
    """ Get a random sampled of a dataset in json coco format. """

    def get_annotations_by_img_id(annotations,
                                  img_id):
        # binary search for image id match in annotations
        l_ptr = 0
        r_ptr = len(annotations)

        idx = -1
        while l_ptr < r_ptr:
            mid = int(r_ptr - l_ptr - 1) // 2 + l_ptr
            current_img_id = annotations[mid]['image_id']
            if current_img_id == target_img_id:
                idx = mid
                break
            elif current_img_id > target_img_id:
                r_ptr = mid - 1
            else:
                l_ptr = mid + 1

        if idx == -1:
            return None

        # look to the left and to the right to get all of the annotations with
        # the same image id
        matching_annotations = []
        ptr = idx
        while(ptr >= 0
            and annotations[ptr]['image_id'] == target_img_id):
            matching_annotations.append(annotations[ptr])
            ptr -= 1
        ptr = idx+1
        while(ptr < len(annotations)
            and annotations[ptr]['image_id'] == target_img_id):
            matching_annotations.append(annotations[ptr])
            ptr += 1

        return matching_annotations
            

    with open(original_json_file, 'r') as file:
        data =json.load(file)

    images = data['images']
    annotations = data['annotations']

    # sort the annotations by image id for binary search later in algorithm
    annotations = sorted(annotations, key=lambda x: x['image_id'])

    if not include_unlabelled_images:
        # remove images without annotations (there are no detections on some images)
        l_ptr = r_ptr = 0
        while r_ptr < len(images):
            target_img_id = images[r_ptr]['id']
            matching_annotations = get_annotations_by_img_id(annotations,
                                                             target_img_id)
            if matching_annotations is not None:
                images[l_ptr], images[r_ptr] = images[r_ptr], images[l_ptr]
                l_ptr += 1
            
            r_ptr += 1
        
        images = images[:l_ptr]
    
    # use a seed for reproducability if specified
    if seed is not None:
        random.seed(seed)
    
    random.shuffle(images)

    sampled_images = images[:sample_size]
    sampled_img_annotations = []
    for img_dict in sampled_images:
        target_img_id = img_dict['id']

        # binary search for image id match in annotations
        l_ptr = 0
        r_ptr = len(annotations)

        idx = -1
        while l_ptr < r_ptr:
            mid = int(r_ptr - l_ptr - 1) // 2 + l_ptr
            current_img_id = annotations[mid]['image_id']
            if current_img_id == target_img_id:
                idx = mid
                break
            elif current_img_id > target_img_id:
                r_ptr = mid - 1
            else:
                l_ptr = mid + 1

        if idx == -1:
            print("Error: No annotations found for this image. Continuing..")
            continue

        # look to the left and to the right to get all of the annotations with
        # the same image id
        ptr = idx
        while(ptr >= 0
              and annotations[ptr]['image_id'] == target_img_id):
            sampled_img_annotations.append(annotations[ptr])
            ptr -= 1
        ptr = idx+1
        while(ptr < len(annotations)
              and annotations[ptr]['image_id'] == target_img_id):
            sampled_img_annotations.append(annotations[ptr])
            ptr += 1

    data['images'] = sampled_images
    data['annotations'] = sampled_img_annotations

    with open(new_json_file, 'w') as file:
        json.dump(data, file)

    return

# TESTING FUNCTIONS #

def print_category_info(json_file,
                        format):
    with open(json_file, 'r') as file:
        data = json.load(file)

    annotations = data['annotations']
    class_labels_present = dict()

    for annotation in annotations:
        if annotation['category_id'] not in class_labels_present:
            class_labels_present[annotation['category_id']] = 1
        else:
            class_labels_present[annotation['category_id']] += 1

    for class_label in class_labels_present:
        if format == 'walaris':
            print(f'{class_label}: {WALARIS_CLASS_LABELS_NUM2NAME[class_label]} - {class_labels_present[class_label]}')
        elif format == 'coco':
            print(f'{class_label}: {COCO_CLASSES_DICT_NUM2NAME[class_label]} - {class_labels_present[class_label]}')
        elif format == 'mmdet_coco':
            print(f'{class_label}: {MMDET_COCO_CLASSES_DICT_NUM2NAME[class_label]} - {class_labels_present[class_label]}')
    return

def visualize_coco_labelled_img(img_path, annotations):
    """Show the bounding boxes of a labelled coco image. Assumes the
    annotations are in coco format.
    
    Parameters
        img_path (str): path to the labelled image
        annotations (list): list of annotation dictionaries corresponding to
         the labelled img
         
    Returns
        None
    """

    # ensure the img_path is only the img file extension
    img_path = img_path.split('/')[-3:]

    base_path = os.environ.get('WALARIS_MAIN_DATA_PATH')

    full_img_path = os.path.join(base_path,
                                 'Images',
                                 img_path[0],
                                 img_path[1],
                                 img_path[2])
    
    # make sure the image can be found on the machine
    assert os.path.exists(full_img_path), "Error: Image not found on machine."

    # read img
    img = cv2.imread(full_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # get the bboxes
    bboxes = []
    class_labels = []
    for annotation in annotations:
        bboxes.append(annotation['bbox'])
        class_labels.append(COCO_CLASSES_DICT_NUM2NAME[annotation['category_id']])

    fig, ax = plt.subplots()
    ax.imshow(img)

    # plot bboxes on image
    view.show_bboxes_plt(bboxes, ax, bbox_format='xywh', labels=class_labels)
    plt.show()

    return

def visualize_coco_ground_truth_dataset(json_file):
    """Randomly visualize images and labels from a dataset in the format of the
    ground truth coco dataset.
    
    """
    with open(json_file, 'r') as file:
        data = json.load(file)

    images = data['images']
    annotations = data['annotations']

    # sort the annotations to easily collect all annotations with a specific
    # instance ID
    annotations = sorted(annotations, key=lambda x: x['image_id'])

    base_image_path = os.environ.get('WALARIS_MAIN_DATA_PATH', )

    while 1:
        # get image information to predict
        idx = np.random.randint(0, len(images))
        img_path_ext = images[idx]['file_name']

        # if image is not saved on machine, continue
        full_path = os.path.join(base_image_path, 
                                 'Images',
                                 img_path_ext)
        if not os.path.exists(full_path):
            print(f'Skipping: Image not found on machine..')
            continue
        target_img_id = images[idx]['id']

        # get all of the annotations with this image id

        # binary search to find annotation
        l_ptr = 0
        r_ptr = len(annotations)

        idx = -1
        while l_ptr < r_ptr:
            mid = int(r_ptr - l_ptr - 1) // 2 + l_ptr
            current_img_id = annotations[mid]['image_id']
            if current_img_id == target_img_id:
                idx = mid
                break
            elif current_img_id > target_img_id:
                r_ptr = mid - 1
            else:
                l_ptr = mid + 1

        if idx == -1:
            print("No annotations found for this image. Continuing..")
            img = cv2.imread(full_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.show()
            continue

        # look to the left and to the right to get all of the annotations with
        # the same image id
        curr_img_annotations = []
        ptr = idx
        while(ptr >= 0
              and annotations[ptr]['image_id'] == target_img_id):
            curr_img_annotations.append(annotations[ptr])
            ptr -= 1
        ptr = idx+1
        while(ptr < len(annotations)
              and annotations[ptr]['image_id'] == target_img_id):
            curr_img_annotations.append(annotations[ptr])
            ptr += 1

        # visualize the image
        visualize_coco_labelled_img(full_path, curr_img_annotations)

def visualize_coco_results(results_json_file,
                           id2img_path_json_file):
    """Randomly visualize images and labels from a dataset in the format of the
    ground truth coco dataset.
    
    """
    with open(results_json_file, 'r') as file:
        data = json.load(file)

    with open(id2img_path_json_file, 'r') as file:
        id2img_path = json.load(file)
    
    annotations = data
    print(len(id2img_path))
    # sort the annotations to easily collect all annotations with a specific
    # instance ID
    annotations = sorted(annotations, key=lambda x: x['image_id'])

    base_image_path = os.environ.get('WALARIS_MAIN_DATA_PATH', )

    while 1:
        # get image information to predict
        idx = np.random.randint(0, len(annotations))
        img_id = str(annotations[idx]['image_id'])
        if img_id not in id2img_path:
            print('ID not in id2img_path...')
            continue
        img_path_ext = id2img_path[img_id]

        # if image is not saved on machine, continue
        full_path = os.path.join(base_image_path, 
                                 'Images',
                                 img_path_ext)
        if not os.path.exists(full_path):
            print(f'Skipping: Image not found on machine..')
            continue
        target_img_id = annotations[idx]['image_id']

        # get all of the annotations with this image id

        # look to the left and to the right to get all of the annotations with
        # the same image id
        curr_img_annotations = []
        ptr = idx
        while(annotations[ptr]['image_id'] == target_img_id):
            curr_img_annotations.append(annotations[ptr])
            ptr -= 1
        ptr = idx+1
        while(annotations[ptr]['image_id'] == target_img_id):
            curr_img_annotations.append(annotations[ptr])
            ptr += 1

        # visualize the image
        visualize_coco_labelled_img(full_path, curr_img_annotations)

if __name__=='__main__':
    walaris_dataset = '/home/grayson/Desktop/Code/WalarisTrainingRepos/Object_Tracking/mmdetection/scripts/json_files/fully_labeled_walaris_id.json'
    coco_format_dataset = '/home/grayson/Desktop/Code/WalarisTrainingRepos/Object_Tracking/mmdetection/scripts/json_files/fully_labeled_coco_id.json'
    cleaned_coco_format_dataset = '/home/grayson/Desktop/Code/WalarisTrainingRepos/Object_Tracking/mmdetection/scripts/json_files/cleaned_coco_format.json'
    random_sample_json_file = '/home/grayson/Desktop/Code/WalarisTrainingRepos/Object_Tracking/mmdetection/scripts/json_files/random_sample_1000_images/fully_labelled_dataset_coco_ids_random_sample.json'

    # convert_format_walaris_to_coco(walaris_dataset,
    #                                coco_format_dataset)

    # classes_to_remove = {
    #     'airplane',
    #     'bird'
    # }

    # clean_json_dataset(coco_format_dataset,
    #                    '/home/grayson/Desktop/Code/WalarisTrainingRepos/Object_Tracking/mmdetection/scripts/json_files/cleaned_coco_format.json',
    #                    'coco',
    #                    classes_to_remove)
    
    # get_random_sample_from_json_file(cleaned_coco_format_dataset,
    #                                  '/home/grayson/Desktop/Code/WalarisTrainingRepos/Object_Tracking/mmdetection/scripts/json_files/random_sample_10000_images/random_sample_1000_images.json',
    #                                  1000,
    #                                  seed=155)
    
    # get_random_sample_from_json_file(coco_format_dataset,
    #                                  random_sample_json_file,
    #                                  sample_size=1000,
    #                                  include_unlabelled_images=True)
    
    # visualize_coco_ground_truth_dataset('/home/grayson/Desktop/Code/WalarisTrainingRepos/Object_Tracking/mmdetection/scripts/json_files/cleaned_coco_format.json')

    visualize_coco_results('/home/grayson/Desktop/Code/WalarisTrainingRepos/Object_Tracking/mmdetection/scripts/json_files/dino_results.json',
                           '/home/grayson/Desktop/Code/WalarisTrainingRepos/Object_Tracking/mmdetection/scripts/json_files/id2img_dino_results.json')