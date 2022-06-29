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
from models.duda_model import *
from utils.duda_utils import *
from utils.dataset import *
from utils.metrics import *

# EVAL TRANSFORMER (NO ATTENTION IN IMAGE ENCODER) MODEL

# Data parameters
data_folder = '/home/gi75qag/dataset/preprocessed_dataset'  # folder with data files saved by create_input_files.py
data_name = 'blocks2D_logos_9_cap_per_img_pair_1_min_word_freq'  # base name shared by data files
word_map_file = '/home/gi75qag/dataset/preprocessed_dataset/WORDMAP_blocks2D_logos_9_cap_per_img_pair_1_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # sets device for model and PyTorch tensors
cudnn.benchmark = True # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# model parameters
captions_per_image = 9
beam_size = 5
batch_size = 1
workers = 1
checkpoint = '/home/gi75qag/models/archive/BEST_checkpoint_blocks2D_logos_9_cap_per_img_pair_1_min_word_freq_DUDA_REG_DOM.pth.tar'  # model checkpoint
checkpoint_name = "BEST_checkpoint_blocks2D_logos_9_cap_per_img_pair_1_min_word_freq_DUDA_REG_DOM.pth.tar"

# Load model
checkpoint = torch.load(checkpoint, map_location=torch.device('cuda'))
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()
image_feature_encoder = checkpoint['image_encoder']
image_feature_encoder = image_feature_encoder.to(device)
image_feature_encoder.eval()

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


# 1. Perform instruction generation step with model and image
# DataLoader
loader = torch.utils.data.DataLoader(
    CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize]), img_return_mode='SPLITTED'),
    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

# Lists to store references (true captions), and hypothesis (prediction) for each image
# If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
# references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
references = list()
hypotheses = list()

alpha_bef = ''
alpha_aft = ''
alphas = ''

# For each image
for i, (image1, image2, caps, caplens, allcaps) in enumerate(
          tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
    
  #early stopping for test reasons
  #if i != example_image_index:
  #  continue
  #if i > example_image_index:
  #  break
    
  k = beam_size

  # Move to GPU device, if available
  image1 = image1.to(device) #  
  image2 = image2.to(device) # 

  # Encode
  image1 = image_feature_encoder(image1) # (1, 14, 14, 2048) 
  image2 = image_feature_encoder(image2) # (1, 14, 14, 2048) 
  enc_image_size = image1.size(1)
  encoder_dim = image1.size(3)

  # We'll treat the problem as having a batch size of k
  image1 = image1.expand(k, enc_image_size, enc_image_size, encoder_dim)  # (k, enc_image_size, enc_image_size, encoder_dim)
  image2 = image2.expand(k, enc_image_size, enc_image_size, encoder_dim)  # (k, enc_image_size, enc_image_size, encoder_dim)

  # Tensor to store top k previous words at each step; now they're just <start>
  k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)   # (k, 1)

  # Tensor to store top k sequences; now they're just <start>
  seqs = k_prev_words # (k, 1)

  # Tensor to store top k sequences' scores; now they're just 0
  top_k_scores = torch.zeros(k, 1).to(device) # (k, 1)

  # Tensor to store top k sequences' alphas; now they're just 1s
  seqs_alpha = torch.ones(k, 1, 3).to(device)  # (k, 1, number_latent_images) => [l_bef,l_diff,l_aft] = 3

  # Lists to store completed sequences and scores
  complete_seqs = list()
  complete_seqs_scores = list()
  complete_seqs_alpha = list()

  # Start decoding
  step = 1

  l_bef, l_aft, alpha_bef, alpha_aft = encoder(image1, image2)
      
  l_diff = torch.sub(l_aft,l_bef)

  l_total = torch.cat([l_bef,l_aft,l_diff],dim=1)

  l_total = decoder.relu(decoder.wd1(l_total))
   
  h_da = torch.zeros(k, decoder.hidden_dim).to(device)  
  c_da = torch.zeros(k, decoder.hidden_dim).to(device)

  h_ds = torch.zeros(k, decoder.hidden_dim).to(device)
  c_ds = torch.zeros(k, decoder.hidden_dim).to(device)

  # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
  while True:
    embeddings = decoder.embedding(k_prev_words).squeeze(1) # (s, embed_dim) 
      
    u_t = torch.cat([l_total, h_ds],dim=1)
    h_da, c_da = decoder.dynamic_att(u_t, (h_da, c_da))

    a_t = decoder.softmax(decoder.wd2(h_da)) # (s, 3)
    alpha = a_t                    

    l_dyn = a_t[:,0].unsqueeze(1)*l_bef + a_t[:,1].unsqueeze(1)*l_aft + a_t[:,2].unsqueeze(1)*l_diff
    c_t = torch.cat([embeddings,l_dyn], dim=1)

    h_ds, c_ds = decoder.decode_step(c_t, (h_ds, c_ds)) # (s, decoder_dim)
   
    scores = decoder.wdc(h_ds) # (s, vocab_size)
    scores = F.log_softmax(scores, dim=1)

    # Add
    scores = top_k_scores.expand_as(scores) + scores # (s, vocab_size)      

    # For the first step, all k points will have the same scores (since same k previous words, h, c)
    if step == 1:
      top_k_scores, top_k_words = scores[0].topk(k, 0, True, True) # (s)
    else:
      # Unroll and find top scores, and their unrolled indices
      top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True) # (s)

    # Convert unrolled indices to actual indices of scores
    prev_word_inds = top_k_words // vocab_size # (s)
    next_word_inds = top_k_words % vocab_size # (s)

    # Add new words to sequences, alphas
    seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1) # (s, step + 1)
    seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, 3)

    # Which sequences are incomplete (didn't reach <end>)?
    incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
    complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

    # Set aside complete sequences
    if len(complete_inds) > 0:
      complete_seqs.extend(seqs[complete_inds].tolist())
      complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
      complete_seqs_scores.extend(top_k_scores[complete_inds])
    k -= len(complete_inds) # reduce beam length accordingly

    # Proceed with incomplete sequences
    if k == 0:
      break
    seqs = seqs[incomplete_inds]
    seqs_alpha = seqs_alpha[incomplete_inds]
    h_ds = h_ds[prev_word_inds[incomplete_inds]]
    c_ds = c_ds[prev_word_inds[incomplete_inds]]
    h_da = h_da[prev_word_inds[incomplete_inds]]
    c_da = c_da[prev_word_inds[incomplete_inds]]
    image1 = image1[prev_word_inds[incomplete_inds]]
    image2 = image2[prev_word_inds[incomplete_inds]]
    l_bef = l_bef[prev_word_inds[incomplete_inds]]
    l_aft = l_aft[prev_word_inds[incomplete_inds]]
    l_diff = l_diff[prev_word_inds[incomplete_inds]]
    l_total = l_total[prev_word_inds[incomplete_inds]]
    top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
    k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

    # Break if things have been going on too long
    if step > 50:
      break
    step += 1
    
  i = complete_seqs_scores.index(max(complete_seqs_scores))
  seq = complete_seqs[i]
  alphas = complete_seqs_alpha[i]

  # References
  img_caps = allcaps[0].tolist()
  img_captions = list(
            map(lambda c: [rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
  references.append(img_captions)

  # Hypotheses
  hypotheses.append([rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
  #hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
  assert len(references) == len(hypotheses)


# EXECUTE EVALUATION ON TEST DATASET
print('USING CHECKPOINT: \n')
print(checkpoint_name)
print('\n')
print('EVALUATING:')

# TARGET DETECTION, LANDMARK DETECTION, SPATIAL DESCRIPTION DETECTION
instruction_parser = SimpleInstructionParser()
# 1. Get target from hypothesis and from reference  => count correct targets found || TEMPORAL REASONING
target_and_landmark = 0
target_and_not_landmark = 0
not_target_and_landmark = 0
not_target_and_not_landmark = 0

target_and_landmark_cases = []
target_and_not_landmark_cases = []
not_target_and_landmark_cases = []
not_target_and_not_landmark_cases = []

for i in range(len(hypotheses)):
    hypothesis_target_block = instruction_parser.get_target_block(hypotheses[i])
    hypothesis_landmark_list = instruction_parser.get_landmarks(hypotheses[i])
    reference_target_block_list = list()
    reference_landmarks = list()
    for j in range(len(references[i])):
        reference_target_block = instruction_parser.get_target_block(references[i][j])
        reference_target_block_list.append(reference_target_block)
        reference_landmark_list = instruction_parser.get_landmarks(references[i][j])
        reference_landmarks += reference_landmark_list
        #print('hypothesis target: ',hypothesis_target_block)
        #print('reference target: ', str(reference_target_block_list))
    landmark_overlap = intersection(hypothesis_landmark_list, reference_landmarks)
    if len(hypothesis_landmark_list) > 0:
        landmark_overlap_ratio =  len(landmark_overlap) / len(hypothesis_landmark_list)
    else:
        landmark_overlap_ratio = 0 
    if ((hypothesis_target_block in reference_target_block_list) and (landmark_overlap_ratio > 0)):
        target_and_landmark +=1
        target_and_landmark_cases.append(i)
    else:
        if ((hypothesis_target_block in reference_target_block_list) and (landmark_overlap_ratio == 0)):
            target_and_not_landmark += 1
            target_and_not_landmark_cases.append(i)
        else:
            if ((hypothesis_target_block not in reference_target_block_list) and (landmark_overlap_ratio > 0)):
                not_target_and_landmark += 1
                not_target_and_landmark_cases.append(i)
            else:
                if ((hypothesis_target_block not in reference_target_block_list) and (landmark_overlap_ratio == 0)):
                    not_target_and_not_landmark +=1
                    not_target_and_not_landmark_cases.append(i)

number_of_hypotheses = len(hypotheses)
number_of_added_partials = target_and_landmark + target_and_not_landmark + not_target_and_landmark + not_target_and_not_landmark
target_and_landmark_correlation = target_and_landmark / number_of_hypotheses
target_and_not_landmark_correlation = target_and_not_landmark / number_of_hypotheses
not_target_and_landmark_correlation = not_target_and_landmark / number_of_hypotheses
not_target_and_not_landmark_correlation = not_target_and_not_landmark / number_of_hypotheses

print('number of hypotheses : ', str(number_of_hypotheses))
print('number of added partials: ', str(number_of_added_partials))
print('\n')
print('target_and_landmark: ', str(target_and_landmark))
print('target_and_not_landmark: ', str(target_and_not_landmark))
print('not_target_and_landmark: ', str(not_target_and_landmark))
print('not_target_and_not_landmark: ', str(not_target_and_not_landmark))
print('\n')
print('target_and_landmark_correlation: ', str(target_and_landmark_correlation))
print('target_and_not_landmark_correlation: ', str(target_and_not_landmark_correlation))
print('not_target_and_landmark_correlation: ', str(not_target_and_landmark_correlation))
print('not_target_and_not_landmark_correlation: ', str(not_target_and_not_landmark_correlation))
print('\n')
print('TARGET AND LANDMARK CASES: ')
print(target_and_landmark_cases)
print('\n')
print('TARGET NOT LANDMARK CASES: ')
print(target_and_not_landmark_cases)
print('\n')
print('NOT TARGET LANDMARK CASES: ')
print(not_target_and_landmark_cases)
print('\n')
print('NOT TARGET NOT LANDMARK CASES: ')
print(not_target_and_not_landmark_cases)
