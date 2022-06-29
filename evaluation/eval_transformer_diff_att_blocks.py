# IMPORTS
import torch.optim
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm
import math
import sys

from utils.dataset import CaptionDataset
from utils.eval_utils import translate, evaluate
from utils.metrics import SimpleInstructionParser, intersection, detokenize_list_of_lists, \
    detokenize_list_of_list_of_lists, formatize_references_and_hypotheses, score

sys.path.append('.')


# CHECKLIST:
# 1. correct model
# 2. correct number of attention heads
# 3. correct test data set loader -> test sets ( nsc test, sc test)
# 4. correct hypothesis objects file path
# 5. correct hypothesis objects file name (png_n)

# sanity check
# assert len(test_image_numbers) == len(hypotheses)
def perform_eval(config, data_loader, model, image_feature_encoder, device):
    # EXECUTE EVALUATION ON TEST DATASET
    print('USING CHECKPOINT: \n')
    print(config.checkpoint_name)
    print('\n')
    print('EVALUATING:')
    references, hypotheses = evaluate(config, data_loader, model, image_feature_encoder, device)

    # Compute metrics
    hypotheses = detokenize_list_of_lists(hypotheses)
    references = detokenize_list_of_list_of_lists(references)

    perform_target_landmark_eval(hypotheses, references)

    references, hypotheses = formatize_references_and_hypotheses(references, hypotheses)
    print(score(references, hypotheses))




def perform_target_landmark_eval(hypotheses, references):
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
        # print('hypothesis target: ',hypothesis_target_block)
        # print('reference target: ', str(reference_target_block_list))
        if hypothesis_target_block in reference_target_block_list:
            correct_target_blocks_detected += 1
    target_detection_ratio = correct_target_blocks_detected / len(hypotheses)
    # 2. Get landmarks from hypothesis and list of landmarks from references => compute overlap || SPATIAL REASONING
    global_landmark_overlap_ratio = 0
    for i in range(len(hypotheses)):
        hypothesis_landmark_list = instruction_parser.get_landmarks(hypotheses[i])
        reference_landmarks = list()
        for j in range(len(references[i])):
            reference_landmark_list = instruction_parser.get_landmarks(references[i][j])
            reference_landmarks += reference_landmark_list
        # print('hypothesis landmarks: ', str(hypothesis_landmark_list))
        # print('reference landmarks: ', str(reference_landmarks))
        landmark_overlap = intersection(hypothesis_landmark_list, reference_landmarks)
        # print('intersection landmarks: ', str(landmark_overlap))
        if len(hypothesis_landmark_list) > 0:
            landmark_overlap_ratio = len(landmark_overlap) / len(hypothesis_landmark_list)
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
        # print('hypothesis spatial descriptions: ', str(hypothesis_spatial_description_list))
        # print('reference spatial descriptions: ', str(reference_spatial_descriptions))
        spatial_descriptions_overlap = intersection(hypothesis_spatial_description_list, reference_spatial_descriptions)
        # print('intersection spatial descriptions: ', str(spatial_descriptions_overlap))
        if len(hypothesis_spatial_description_list) > 0:
            spatial_descriptions_overlap_ratio = len(spatial_descriptions_overlap) / len(
                hypothesis_spatial_description_list)
        else:
            spatial_descriptions_overlap_ratio = 0
        global_spatial_description_overlap_ratio += spatial_descriptions_overlap_ratio
    spatial_description_detection_ratio = global_spatial_description_overlap_ratio / len(hypotheses)
    print('target_detection_ratio: ', str(target_detection_ratio))
    print('landmark_detection_ratio: ', str(landmark_detection_ratio))
    print('spatial_description_detection_ratio: ', str(spatial_description_detection_ratio))
