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
dictionary_path_absolute = '/home/gi75qag/models/knn_nearest_neighbor_dict_blocks2D_logos_9_cap_per_img_pair_1_min_word_freq.txt' # absolute path of the nearest neighbor dictionary file
image_pair_index = 0

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
  #if i > 1:
  #  break
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

#print(hypotheses)
#print(references)

# TARGET DETECTION, LANDMARK DETECTION, SPATIAL DESCRIPTION DETECTION
instruction_parser = SimpleInstructionParser()
# 1. Get target from hypothesis and from reference  => count correct targets found || TEMPORAL REASONING
correct_target_blocks_detected = 0
for i in range(len(hypotheses)):
  hypothesis_target_block = instruction_parser.get_target_block(hypotheses[i])
  reference_target_block_list = list()
  for j in range(len(references[i])):
    reference_target_block = instruction_parser.get_target_block(references[i][j])
    reference_target_block_list.append(reference_target_block)
  #print('hypothesis target: ',hypothesis_target_block)
  #print('reference target: ', str(reference_target_block_list))
  if hypothesis_target_block in reference_target_block_list:
    correct_target_blocks_detected +=1

target_detection_ratio = correct_target_blocks_detected / len(hypotheses)
# 2. Get landmarks from hypothesis and list of landmarks from references => compute overlap || SPATIAL REASONING
global_landmark_overlap_ratio = 0
for i in range(len(hypotheses)):
  hypothesis_landmark_list = instruction_parser.get_landmarks(hypotheses[i])
  reference_landmarks = list()
  for j in range(len(references[i])):
    reference_landmark_list = instruction_parser.get_landmarks(references[i][j])
    reference_landmarks += reference_landmark_list
  #print('hypothesis landmarks: ', str(hypothesis_landmark_list))
  #print('reference landmarks: ', str(reference_landmarks))
  landmark_overlap = intersection(hypothesis_landmark_list, reference_landmarks)
  #print('intersection landmarks: ', str(landmark_overlap))
  if len(hypothesis_landmark_list) > 0:
    landmark_overlap_ratio =  len(landmark_overlap) / len(hypothesis_landmark_list)
  else:
    landmark_overlap_ratio = 0 
  global_landmark_overlap_ratio += landmark_overlap_ratio

landmark_detection_ratio = global_landmark_overlap_ratio / len(hypotheses)

# 3. Get spatial descriptions from hypothesis and list of spatial descriptions from references => compute overlap || SPATIAL REASONING
global_spatial_description_overlap_ratio = 0
for i in range(len(hypotheses)):
  hypothesis_spatial_description_list = instruction_parser.get_spatial_descriptions(hypotheses[i])
  reference_spatial_descriptions = list()
  for j in range(len(references[i])):
    reference_spatial_descriptions_list = instruction_parser.get_spatial_descriptions(references[i][j])
    reference_spatial_descriptions += reference_spatial_descriptions_list
  #print('hypothesis spatial descriptions: ', str(hypothesis_spatial_description_list))
  #print('reference spatial descriptions: ', str(reference_spatial_descriptions))  
  spatial_descriptions_overlap = intersection(hypothesis_spatial_description_list, reference_spatial_descriptions)
  #print('intersection spatial descriptions: ', str(spatial_descriptions_overlap))
  if len(hypothesis_spatial_description_list) > 0:
    spatial_descriptions_overlap_ratio =  len(spatial_descriptions_overlap) / len(hypothesis_spatial_description_list) 
  else:
    spatial_descriptions_overlap_ratio = 0
  global_spatial_description_overlap_ratio += spatial_descriptions_overlap_ratio

spatial_description_detection_ratio = global_spatial_description_overlap_ratio / len(hypotheses)

print('target_detection_ratio: ', str(target_detection_ratio))
print('landmark_detection_ratio: ', str(landmark_detection_ratio))
print('spatial_description_detection_ratio: ', str(spatial_description_detection_ratio))

# Compute metrics
hypotheses = detokenize_list_of_lists(hypotheses)
references = detokenize_list_of_list_of_lists(references)
references, hypotheses = formatize_references_and_hypotheses(references, hypotheses)
print(score(references, hypotheses)) 