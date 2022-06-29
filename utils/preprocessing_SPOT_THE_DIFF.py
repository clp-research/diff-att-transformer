# IMPORTS
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import h5py
import json
import torch
from skimage.transform import resize as imresize
from skimage.io import imread
from skimage import color
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import sys
sys.path.append('.')
#from cl_1.class_1 import Printer


# PREPROCESSING FUNCTIONS

# Create dataset files function 
def create_input_files(dataset, train_captions_json_path, val_captions_json_path, test_captions_json_path, 
                       train_image_folder, val_image_folder, test_image_folder, captions_per_image_pair, min_word_freq, output_folder,
                       max_len=100):
    """
    Creates input files for training, validation, and test data.
    :param dataset: name of dataset, one of 'blocks2D_logos', 'blocks2D_digits', 'blocks3D_blank', 'blocks3D_logos'
    :param train_captions_json_path: path of Train Captions JSON file with captions
    :param val_captions_json_path: path of Validation Captions JSON file captions
    :param test_captions_json_path: path of Test Captions JSON file captions
    :param train_image_folder: folder with downloaded images for training
    :param val_image_folder: folder with downloaded images for validation
    :param test_image_folder: folder with downloaded images for testing
    :param captions_per_image_pair: number of captions to sample per image pair
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'SPOT_THE_DIFF'}

    # Storage Variables
    train_image_pairs_paths = []
    train_image_pairs_captions = []
    val_image_pairs_paths = []
    val_image_pairs_captions = []
    test_image_pairs_paths = []
    test_image_pairs_captions = []
    word_freq = Counter()
    
    with open(train_captions_json_path) as train_file_json:
        train_captions_json_data = json.load(train_file_json)
        
    with open(val_captions_json_path) as val_file_json:
        val_captions_json_data = json.load(val_file_json)
        
    with open(test_captions_json_path) as test_file_json:
        test_captions_json_data = json.load(test_file_json)

    # TRAINING DATA
    # Read Train Captions JSON
    train_set = train_captions_json_data
    
    for train_image in train_set:
        captions = []
        train_image_number = str(train_image['img_id'])
        train_image_number_filename_before = train_image_number + '.png'
        img_before_path = os.path.join(train_image_folder, train_image_number_filename_before)
        train_image_number_filename_after = train_image_number + '_2.png'
        img_after_path = os.path.join(train_image_folder, train_image_number_filename_after) 
        for single_caption in train_image['sentences']:
            #apply text preprocessing for each caption: lowercasing, tokenization
            train_image_pair_caption = single_caption.lower().split()
            # Update word frequency
            word_freq.update(train_image_pair_caption)
            if len(train_image_pair_caption) <= max_len:
                captions.append(train_image_pair_caption)
        train_image_pairs_captions.append(captions)
        train_image_pairs_paths.append([img_before_path, img_after_path])
    
    # VALIDATION DATA
    # Read Test Captions JSON
    val_set = val_captions_json_data
    
    for val_image in val_set:
        captions = []
        val_image_number = str(val_image['img_id'])
        val_image_number_filename_before = val_image_number + '.png'
        img_before_path = os.path.join(val_image_folder, val_image_number_filename_before)
        val_image_number_filename_after = val_image_number + '_2.png'
        img_after_path = os.path.join(val_image_folder, val_image_number_filename_after) 
        for single_caption in val_image['sentences']:
            #apply text preprocessing for each caption: lowercasing, tokenization
            val_image_pair_caption = single_caption.lower().split()
            # Update word frequency
            word_freq.update(val_image_pair_caption)
            if len(val_image_pair_caption) <= max_len:
                captions.append(val_image_pair_caption)
        val_image_pairs_captions.append(captions)
        val_image_pairs_paths.append([img_before_path, img_after_path])
    
    # TEST DATA
    # Read Test Captions JSON
    test_set = test_captions_json_data
    
    for test_image in test_set:
        captions = []
        test_image_number = str(test_image['img_id'])
        test_image_number_filename_before = test_image_number + '.png'
        img_before_path = os.path.join(test_image_folder, test_image_number_filename_before)
        test_image_number_filename_after = test_image_number + '_2.png'
        img_after_path = os.path.join(test_image_folder, test_image_number_filename_after) 
        for single_caption in test_image['sentences']:
            #apply text preprocessing for each caption: lowercasing, tokenization
            test_image_pair_caption = single_caption.lower().split()
            # Update word frequency
            word_freq.update(test_image_pair_caption)
            if len(test_image_pair_caption) <= max_len:
                captions.append(test_image_pair_caption)
        test_image_pairs_captions.append(captions)
        test_image_pairs_paths.append([img_before_path, img_after_path])
      
    #print(test_image_pairs_paths)
    #print('EXITING script.')
    #sys.exit()  
      
    # Sanity check
    print('Sanity Check lengths ...')
    assert len(train_image_pairs_paths) == len(train_image_pairs_captions)
    assert len(val_image_pairs_paths) == len(val_image_pairs_captions)
    assert len(test_image_pairs_paths) == len(test_image_pairs_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image_pair) + '_cap_per_img_pair_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image pair, save image pairs to separate HDF5 file as single image with size (3, 224, 224), and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_pairs_paths, train_image_pairs_captions, 'TRAIN'), 
                                   (val_image_pairs_paths, val_image_pairs_captions, 'VAL'),
                                   (test_image_pairs_paths, test_image_pairs_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_BEFORE_' + base_filename + '.hdf5'), 'a') as h1, h5py.File(os.path.join(output_folder, split + '_IMAGES_AFTER_' + base_filename + '.hdf5'), 'a') as h2:
            # Make a note of the number of captions we are sampling per image
            h1.attrs['captions_per_image_pair'] = captions_per_image_pair
            h2.attrs['captions_per_image_pair'] = captions_per_image_pair

            # Create dataset inside HDF5 file to store images
            images_before = h1.create_dataset('images_before', (len(impaths), 3, 224, 224), dtype='float')
            images_after = h2.create_dataset('images_after', (len(impaths), 3, 224, 224), dtype='float')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image_pair:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image_pair - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image_pair)

                # Sanity check
                assert len(captions) == captions_per_image_pair

                # Read image pairs and save them into intended HDF5 file
                img_before = imread(impaths[i][0])
                #img_before_rgb = color.rgba2rgb(img_before)
                #img_before = imresize(img_before_rgb, (224, 224))
                assert img_before.shape == (224, 224, 3)
                img_before = img_before.transpose(2,0,1)
                assert img_before.shape == (3, 224, 224)
                assert np.max(img_before) <= 255
                # Save image_before to HDF5 file
                images_before[i] = img_before
                
                img_after = imread(impaths[i][1])
                #img_after_rgb = color.rgba2rgb(img_after)
                #img_after = imresize(img_after_rgb, (224, 224))
                assert img_after.shape == (224, 224, 3)
                img_after = img_after.transpose(2,0,1)
                assert img_after.shape == (3, 224, 224)
                assert np.max(img_after) <= 255
                # Save image_before to HDF5 file
                images_after[i] = img_after

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images_before.shape[0] * captions_per_image_pair == len(enc_captions) == len(caplens)
            assert images_after.shape[0] * captions_per_image_pair == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j) 


# PREPROCESSING EXECUTION

# FILE PATHS
base_dir = '/home/gi75qag' # Add your respective path here
# train dataset  
rel_train_captions_path = 'papers/ACL_2021_LSTM_Transformers_for_Change_Captioning/datasets/spot_the_diff_dataset/train.json'
train_captions_json_path = os.path.join(base_dir, rel_train_captions_path)
rel_train_image_path = 'papers/ACL_2021_LSTM_Transformers_for_Change_Captioning/datasets/spot_the_diff_dataset/resized_images/resized_images'
train_image_folder = os.path.join(base_dir, rel_train_image_path)
# validation dataset 
rel_validation_captions_path = 'papers/ACL_2021_LSTM_Transformers_for_Change_Captioning/datasets/spot_the_diff_dataset/val.json'
val_captions_json_path = os.path.join(base_dir, rel_validation_captions_path)
rel_validation_image_path = 'papers/ACL_2021_LSTM_Transformers_for_Change_Captioning/datasets/spot_the_diff_dataset/resized_images/resized_images'
val_image_folder = os.path.join(base_dir, rel_validation_image_path)
# test dataset 
rel_test_captions_path = 'papers/ACL_2021_LSTM_Transformers_for_Change_Captioning/datasets/spot_the_diff_dataset/test.json'
test_captions_json_path = os.path.join(base_dir, rel_test_captions_path)
rel_test_image_path = 'papers/ACL_2021_LSTM_Transformers_for_Change_Captioning/datasets/spot_the_diff_dataset/resized_images/resized_images'
test_image_folder = os.path.join(base_dir, rel_test_image_path)
# output folder to save wordmap and HDF5 file
rel_output_path = 'papers/ACL_2021_LSTM_Transformers_for_Change_Captioning/datasets/spot_the_diff_dataset/preprocessed_dataset'
output_folder = os.path.join(base_dir, rel_output_path)

# IMAGE PROPERTIES
captions_per_image_pair = 3

# CAPTION PROPERTIES
min_word_freq = 1

create_input_files('SPOT_THE_DIFF', train_captions_json_path, val_captions_json_path, test_captions_json_path, 
                       train_image_folder, val_image_folder, test_image_folder, captions_per_image_pair, min_word_freq, output_folder,
                       max_len=100) 