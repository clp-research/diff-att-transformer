# IMPORTS
import torch
import torch.nn.functional as F
import math
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import json
import sys
sys.path.append('.')
from models.transformer_self_att_model import *
from utils.transformer_utils import *
from utils.dataset import *
from utils.metrics import *

# EVAL TRANSFORMER (SELF ATTENTION IN IMAGE ENCODER) MODEL

# Data parameters
data_folder = '/home/gi75qag/dataset/preprocessed_dataset'  # folder with data files saved by create_input_files.py
data_name = 'blocks2D_logos_9_cap_per_img_pair_1_min_word_freq'  # base name shared by data files
word_map_file = '/home/gi75qag/dataset/preprocessed_dataset/WORDMAP_blocks2D_logos_9_cap_per_img_pair_1_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # sets device for model and PyTorch tensors
cudnn.benchmark = True # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# model parameters
captions_per_image = 9
batch_size = 1
workers = 1
checkpoint = '/home/gi75qag/models/archive/BEST_transformer_self_att_8_FINAL_checkpoint.pth.tar'  # model checkpoint
checkpoint_name = "BEST_transformer_self_att_8_FINAL_checkpoint"

# Load model
checkpoint = torch.load(checkpoint, map_location=torch.device('cuda'))
model = checkpoint['model'].to(device)
model.eval()
image_feature_encoder = checkpoint['image_encoder'].to(device)
image_feature_encoder.eval()
beam_size = 5
nb_heads = 8 # number of attention heads on IMAGE used in the model -> important for figuring out visual word/sentence size

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def get_key(dict_, value):
  return [k for k, v in dict_.items() if v == value]

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

def translate(img_before, img_after , beam_size=5, length_norm_coefficient=0.6):
    """
    Translates a source language sequence to the target language, with beam search decoding.
    :param source_sequence: the source language sequence, either a string or tensor of bpe-indices
    :param beam_size: beam size
    :param length_norm_coefficient: co-efficient for normalizing decoded sequences' scores by their lengths
    :return: the best hypothesis, and all candidate hypotheses
    """
    with torch.no_grad():
        # Beam size
        k = beam_size

        # Minimum number of hypotheses to complete
        n_completed_hypotheses = min(k, 10)

        # Vocab size
        vocab_size = len(word_map)

        # Move to GPU, if available
        img_before = img_before.to(device)
        img_after = img_after.to(device)

        # Extract Image Features/Image Encoder
        img_before = image_feature_encoder(img_before) # == source_sequences
        img_after = image_feature_encoder(img_after) # == source_sequences
        image_embedding_attention_heads = nb_heads
        source_sequence_lengths = torch.LongTensor([image_embedding_attention_heads]).to(device) # (1) source sequence length specifies the number of created visual words = nb_heads
        source_sequence_lengths = source_sequence_lengths.to(device) # (N) => in this case it's just one so N=1

        # Encode
        encoder_sequences, img_before_alphas, img_after_alphas = model.encoder(img_before, img_after,
                                          encoder_sequence_lengths=source_sequence_lengths)  # (1, source_sequence_length, d_model)

        # Our hypothesis to begin with is just <start>
        hypotheses = torch.LongTensor([[word_map['<start>']]]).to(device)  # (1, 1)
        hypotheses_lengths = torch.LongTensor([hypotheses.size(1)]).to(device)  # (1)

        # Tensor to store hypotheses' scores; now it's just 0
        hypotheses_scores = torch.zeros(1).to(device)  # (1)

        # Lists to store completed hypotheses and their scores
        completed_hypotheses = list()
        completed_hypotheses_scores = list()

        # Start decoding
        step = 1

        # Assume "s" is the number of incomplete hypotheses currently in the bag; a number less than or equal to "k"
        # At this point, s is 1, because we only have 1 hypothesis to work with, i.e. "<BOS>"
        while True:
            s = hypotheses.size(0)
            decoder_sequences = model.decoder(decoder_sequences=hypotheses,
                                              decoder_sequence_lengths=hypotheses_lengths,
                                              encoder_sequences=encoder_sequences.repeat(s, 1, 1),
                                              encoder_sequence_lengths=source_sequence_lengths.repeat(
                                                  s))  # (s, step, vocab_size)

            # Scores at this step
            scores = decoder_sequences[:, -1, :]  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=-1)  # (s, vocab_size)

            # Add hypotheses' scores from last step to scores at this step to get scores for all possible new hypotheses
            scores = hypotheses_scores.unsqueeze(1) + scores  # (s, vocab_size)

            # Unroll and find top k scores, and their unrolled indices
            top_k_hypotheses_scores, unrolled_indices = scores.view(-1).topk(k, 0, True, True)  # (k)

            # Convert unrolled indices to actual indices of the scores tensor which yielded the best scores
            prev_word_indices = unrolled_indices // vocab_size  # (k)
            next_word_indices = unrolled_indices % vocab_size  # (k)

            # Construct the the new top k hypotheses from these indices
            top_k_hypotheses = torch.cat([hypotheses[prev_word_indices], next_word_indices.unsqueeze(1)],
                                         dim=1)  # (k, step + 1)

            # Which of these new hypotheses are complete (reached <EOS>)?
            complete = next_word_indices == word_map['<end>'] # (k), bool

            # Set aside completed hypotheses and their scores normalized by their lengths
            # For the length normalization formula, see
            # "Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation"
            completed_hypotheses.extend(top_k_hypotheses[complete].tolist())
            norm = math.pow(((5 + step) / (5 + 1)), length_norm_coefficient)
            completed_hypotheses_scores.extend((top_k_hypotheses_scores[complete] / norm).tolist())

            # Stop if we have completed enough hypotheses
            if len(completed_hypotheses) >= n_completed_hypotheses:
                break

            # Else, continue with incomplete hypotheses
            hypotheses = top_k_hypotheses[~complete]  # (s, step + 1)
            hypotheses_scores = top_k_hypotheses_scores[~complete]  # (s)
            hypotheses_lengths = torch.LongTensor(hypotheses.size(0) * [hypotheses.size(1)]).to(device)  # (s)

            # Stop if things have been going on for too long
            if step > 50:
                break
            step += 1

        # If there is not a single completed hypothesis, use partial hypotheses
        if len(completed_hypotheses) == 0:
            completed_hypotheses = hypotheses.tolist()
            completed_hypotheses_scores = hypotheses_scores.tolist()

        # Get attention weights of the encoder-decoder-multi-head-attention layer 
        encoder_decoder_mh_att_weights = model.decoder.decoder_layers[0][1].att_weights # get the enc-dec attention weights of the last decoder layer

        # Decode the hypotheses
        all_hypotheses = list()
        for i, h in enumerate(list(
                map(lambda c: [rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                completed_hypotheses))):
            all_hypotheses.append({"hypothesis": h, "score": completed_hypotheses_scores[i]})

        # Find the best scoring completed hypothesis
        i = completed_hypotheses_scores.index(max(completed_hypotheses_scores))
        best_hypothesis = all_hypotheses[i]["hypothesis"]

        return best_hypothesis, all_hypotheses, img_before_alphas, encoder_decoder_mh_att_weights #img_after_alphas


def evaluate(beam_size):
  with torch.no_grad():
    # DataLoader
    loader = torch.utils.data.DataLoader(
    CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize]), img_return_mode='SPLITTED'),
    batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    for i, (img_before, img_after, caps, caplens, allcaps) in enumerate(
          tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
      
      # early stopping for testing reasons
      #if i > 10:
      #  break
      
      # get best hypothesis from translation
      best_hypothesis, _, _, _ = translate(img_before, img_after,
                                    beam_size=beam_size,
                                    length_norm_coefficient=0.6)
      # References
      img_caps = allcaps[0].tolist()
      img_captions = list(
            map(lambda c: [rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
      references.append(img_captions)

      #best_hypothesis
      # Hypotheses
      hypotheses.append([w for w in best_hypothesis if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
      assert len(references) == len(hypotheses)

  return references, hypotheses


# EXECUTE EVALUATION ON TEST DATASET
print('USING CHECKPOINT: \n')
print(checkpoint_name)
print('\n')
print('EVALUATING:')
beam_size = 5
references, hypotheses = evaluate(beam_size)

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