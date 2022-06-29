# IMPORTS
from nltk.translate.bleu_score import corpus_bleu
from torch.autograd import Variable
import random
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
import json
import sys
import os
sys.path.append('.')
from utils.dataset import CaptionDataset
from models.nearest_neighbor_baseline_model import *


# TRAIN NEAREST NEIGHBOR BASELINE

# PARAMETERS
# Data parameters
data_folder = '/home/users/sadler/data/blockworld_pre'  # folder with data files saved by create_input_files.py
data_name = 'blocks2D_logos_9_cap_per_img_pair_1_min_word_freq'  # base name shared by data files
dictionary_path_absolute = '/home/users/sadler/cache/052_block_instruct_transformer/models/knn_nearest_neighbor_dict_blocks2D_logos_9_cap_per_img_pair_1_min_word_freq.txt' # absolute path of the nearest neighbor dictionary file

# Model parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
model_batch_size = 4
test_batch_size = 1
workers = 1  # for data-loading; right now, only 1 works with h5py

# Load Model and Test Dataset
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
# Load Model Dataset and Test Dataset
model_dataset = CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize]))
test_dataset =  CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize]))
model_dataloader = torch.utils.data.DataLoader(
  model_dataset,
  batch_size=model_batch_size, shuffle=False, num_workers=workers, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(
  test_dataset,
  batch_size=test_batch_size, shuffle=False, num_workers=workers, pin_memory=True)

# Create Image_Encoder
image_encoder = Encoder()
nearest_neighbor_classifier = NearestNeighbor(model_dataloader, image_encoder)

# initialize empty dictionary of nearest neighbours
#write neatest neighbour in dictionary like: nearest_neighbours = {test_item_index:'nearest_neighbour_model_index'}
nearest_neighbors_dict = {}
#load existing dictionary from file
if(os.path.isfile(dictionary_path_absolute)):
    print('***Found Existing Nearest Neighbor Dictionary File***')
    with open(dictionary_path_absolute, "r") as nearest_neighbor_dict_file:
        nearest_neighbors_dict = json.load(nearest_neighbor_dict_file)
else:
    print('***Creating Nearest Neighbor Dictionary File***')
    file = open(dictionary_path_absolute, "w") 
    file.write("") 
    file.close() 

last_test_index = len(nearest_neighbors_dict)
print(last_test_index)
print(nearest_neighbors_dict)

# Get nearest neighbor for each image pair in test data set and safe it into dictionary
for i in range(last_test_index, len(test_dataloader.dataset)):
  print('Processing Test Image Index: ', str(i) + '/' + str(len(test_dataloader.dataset)-1))
  #if i >= 1: 
  #  break
  # get test_image from index i
  test_image, _, _, _ = test_dataset.__getitem__(i)
  test_image = test_image.unsqueeze_(0)
  # find nearest neighbor
  nearest_neighbor_index, _, _, _ = nearest_neighbor_classifier.apply(test_image)
  # append nearest neighbor do dictionary
  nearest_neighbors_dict[i] = nearest_neighbor_index.item()
  # save nearest neighbor dictionary into file
  with open(dictionary_path_absolute, "w") as nearest_neighbor_dict_file:
     nearest_neighbor_dict_file.write(json.dumps(nearest_neighbors_dict))


print('Final Nearest Neighbor Dictionary:')
print(nearest_neighbors_dict)