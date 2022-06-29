# IMPORTS
import os
import sys
import json
sys.path.append('.')
from utils.dataset import CaptionDataset
from utils.metrics import *
from models.nearest_neighbor_baseline_model import *

# EVAL NEAREST NEIGHBOR BASELINE

# PARAMETERS
# Data parameters
data_folder = '/home/gi75qag/dataset/preprocessed_dataset'  # folder with data files saved by create_input_files.py
data_name = 'blocks2D_logos_9_cap_per_img_pair_1_min_word_freq'  # base name shared by data files
dictionary_path_absolute = '/home/gi75qag/models/archive/knn_nearest_neighbor_dict_blocks2D_logos_9_cap_per_img_pair_1_min_word_freq.txt' # absolute path of the nearest neighbor dictionary file
image_pair_index = 513

# Load Train Dataset                                                               
train_dataset = CaptionDataset(data_folder, data_name, 'TRAIN', transform=None, img_return_mode='SPLITTED')
test_dataset = CaptionDataset(data_folder, data_name, 'TEST', transform=None, img_return_mode='SPLITTED')

# Load word map
word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
with open(word_map_file, 'r') as j:
  word_map = json.load(j)

# Load reverse wordmap from index back to word
rev_word_map = {v: k for k, v in word_map.items()}  

# Get pre-computed mapping to nearest neighbor from dictionary 
nearest_neighbors_dict = {}
#load existing dictionary from file
with open(dictionary_path_absolute, "r") as nearest_neighbor_dict_file:
     nearest_neighbors_dict = json.load(nearest_neighbor_dict_file)

print('***Found Nearest Neighbor Dictionary***')
#print(nearest_neighbors_dict)

# COMPUTE BLEU-4, METEOR, CIDEr
# Lists to store references (true captions), and hypothesis (prediction) for each image
# If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
# references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
references = list()
hypotheses = list()
# Iterate over all test_image - nearest_neighbor_pairs
for i in range (len(nearest_neighbors_dict)):
  if i != image_pair_index:
    continue
  if i > image_pair_index:
    break
  nearest_neighbor_index = nearest_neighbors_dict.get(str(i))
  test_image_index = i
  # Hypotheses
  _, _, caption, _ = train_dataset.__getitem__(nearest_neighbor_index) # nearest neighbor
  caption = caption.tolist() # turn tensor into list
  hypotheses.append([rev_word_map[w] for w in caption if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]) # get rid of pads + turn back to words
  # References
  _, _, _, _, all_captions = test_dataset.__getitem__(test_image_index) # gold standard, all_captions.shape = [9,102]
  img_caps = all_captions.tolist() # turn tensor of tensors into list of tensors
  img_captions = list(
              map(lambda c: [rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}], # get rid of pads + turn back to words
                  img_caps))  # remove <start> and pads
  references.append(img_captions)
  assert len(references) == len(hypotheses)

print('hypothesis: ')
print(hypotheses)
print('references: ')
print(references)