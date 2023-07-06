from argparse import ArgumentParser
from mmdet.apis import init_detector, inference_detector
import scripts.utils as utils
import os
from tqdm import tqdm
from data_processing import labels, view
from data_processing.labels import WALARIS_CLASS_LABELS_NUM2NAME
from datetime import datetime
import random
import json

"""This script was used to automatically label the images in the fully labelled
subset of the Tarsier Dataset.

"""

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--model_type', 
        default='dino', 
        type=str,
        choices=['dino', 'yolox', 'rtmdet'],
        help='Model type to use for inference')
    parser.add_argument(
        '--annotation_format',
        type=str,
        choices=['walaris', 'coco'],          
        help='Format for annotation .json file.')
    parser.add_argument(
        '--save_folder',
        type=str,
        default='/home/grayson/Desktop/Code/WalarisTrainingRepos/Object_Tracking/mmdetection/scripts/json_files',
        help='Json file to save labels to.')
    parser.add_argument(
        '--resume',
        '-r',
        default=False,
        help='Specify a json result file to resume training from.'
    )
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args

def load_model(args):
    base_path = '/home/grayson/Desktop/Code/WalarisTrainingRepos/Object_Tracking/mmdetection'
    if args.model_type == 'dino':
        config_path = os.path.join(base_path, 'configs/dino/dino-5scale_swin-l_8xb2-36e_coco.py')
        checkpoint_path =  os.path.join(base_path, 'checkpoints/dino/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth')
    elif args.model_type == 'yolox':
        config_path = os.path.join(base_path, 'configs/yolox/yolox_x_8xb8-300e_coco.py')
        checkpoint_path = os.path.join(base_path, 'checkpoints/rtmdet/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth')
    elif args.model_type == 'rtmdet':
        config_path = os.path.join(base_path, 'configs/rtmdet/rtmdet_l_8xb32-300e_coco.py')
        checkpoint_path = os.path.join(base_path, 'checkpoints/rtmdet/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth')
    else:
        print('Error! Model type not valid..')

    return init_detector(config_path, checkpoint_path, args.device)

def main(args):

    coco_format_json_file = '/home/grayson/Documents/model_training/Tarsier_Main_Dataset/Labels_NEW/day/train_COCO_18_classes/20200731_22C_2_walaris_day_train_coco.json'
    
    now = datetime.now()

    new_label_save_file = os.path.join(args.save_folder, 'results', f'{args.model_type}_fully_labelled_day_train_results.json')
    id2img_path_save_file = os.path.join(args.save_folder, 'id2img_path', 'id2img_day_train_results.json')
    save_every = 50000

    # build the model from a config file and a checkpoint file
    model = load_model(args)

    # specify which classes you want the model to detect
    classes_to_predict = {
        'person',
        'bicycle',
        'boat',
        'bus',
        'car',
        'motorcycle',
        'train',
        'truck'
    }
    
    with open(coco_format_json_file, 'r') as file:
        data = json.load(file)

    images = data['images']
    categories = data['categories']
    annotations = data['annotations']

    # get set of all img paths to test and id2img_path dictionary

    img_paths, id2img_path = utils.get_img_path_info(images)

    # save id2img dictionary to json file
    with open(id2img_path_save_file, 'w') as file:
        json.dump(id2img_path, file)

    class_labels_in_dataset = {
        "uav": 0,
        "airplane": 0, 
        "bicycle": 0,
        "bird": 0,
        "boat": 0,
        "bus": 0,
        "car": 0,
        "cat": 0,
        "cow": 0,
        "dog": 0,
        "horse": 0,
        "motorcycle": 0,
        "person": 0,
        "traffic_light": 0,
        "train": 0,
        "truck": 0,
        "ufo": 0,
        "helicopter": 0,
        "phantom": 0,
        "mavic": 0,
        "spark": 0,
        "inspire": 0
    }

    # remove any annotations that are included in the classes_to_predict set
    new_annotations = []
    for annotation in tqdm(annotations):
        class_name = WALARIS_CLASS_LABELS_NUM2NAME[annotation['category_id']]
        class_labels_in_dataset[class_name] += 1
        if class_name in classes_to_predict:
            continue
        new_annotations.append(annotation)

    print('Class labels found in dataset...')
    print(class_labels_in_dataset)
    print('\n\n')

    # zero the class label count
    for class_name in class_labels_in_dataset:
        class_labels_in_dataset[class_name] = 0

    # verify that only the correct labels were removed
    for annotation in tqdm(new_annotations):
        class_name = WALARIS_CLASS_LABELS_NUM2NAME[annotation['category_id']]
        class_labels_in_dataset[class_name] += 1

    print('Class labels found after removal...')
    print(class_labels_in_dataset)
    
    # free up memory from previous annotations list
    # del annotations[:]

    # set base path based on main dataset path environment variable
    base_image_path = os.path.join(os.environ.get('WALARIS_MAIN_DATA_PATH'),
                                   'Images')
    
    # loop through each image, make prediction, save annotation in the
    # new_annotations lists
    # loop through all images and get the detection results
    for idx in tqdm(range(len(images))):
        image = images[idx]
        img_path_ext = image['file_name']
        img_id = image['id']
        full_img_path = os.path.join(base_image_path, img_path_ext)

        # get detection predictions
        result = inference_detector(model, full_img_path)

        # refine the results based on confidence score threshold
        bboxes, class_labels, scores = utils.get_results_over_thr(result, 
                                                                  args.score_thr)

        # convert to list from tensor
        bboxes = bboxes.tolist()
        class_labels = class_labels.tolist()
        scores = scores.tolist()

        result = bboxes, class_labels, scores

        # remove any detected classes not in the specified classes to detect
        result = utils.remove_unspecified_class_predictions(result,
                                                            classes_to_predict,
                                                            pred_format='mmdet')

        # convert the results to coco_format
        result = utils.result_mmdet_2_coco_format(result, 
                                                  args.score_thr,
                                                  img_id,
                                                  label_convention='walaris')
        
        # add all new predictions to overal result list
        new_annotations += result

        if (idx) % save_every == 0:
            data['annotations'] = new_annotations
            with open(new_label_save_file, 'w') as file:
                json.dump(data, file)

    # save the new labels file
    data['annotations'] = new_annotations
    with open(new_label_save_file, 'w') as file:
        json.dump(data, file)

import signal
import time
import sys

def run_program():
    while True:
        time.sleep(1)
        print("a")

def exit_gracefully(signum, frame):
    # restore the original signal handler as otherwise evil things will happen
    # in raw_input when CTRL+C is pressed, and our signal handler is not re-entrant
    signal.signal(signal.SIGINT, original_sigint)

    try:
        if input("\nReally quit? (y/n)> ").lower().startswith('y'):
            sys.exit(1)

    except KeyboardInterrupt:
        print("Ok ok, quitting")
        sys.exit(1)

    # restore the exit gracefully handler here    
    signal.signal(signal.SIGINT, exit_gracefully)

if __name__ == '__main__':
    # store the original SIGINT handler
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, exit_gracefully)
    args = parse_args()
    main(args)