from data_processing import view

if __name__=='__main__':
    coco_json_file = '/home/grayson/Documents/model_training/Tarsier_Main_Dataset/Labels_NEW/day/train_COCO_18_classes_test_2/anil_label_converter_train.json'
    view.visualize_coco_ground_truth_dataset('/home/grayson/Desktop/Code/WalarisTrainingRepos/Object_Tracking/mmdetection/scripts/json_files/results/dino_fully_labelled_day_train_results_1.json',
                                             'walaris')



