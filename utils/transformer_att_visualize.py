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
from models.transformer_diff_att_model import *
from utils.transformer_utils import *
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
batch_size = 1
workers = 1
checkpoint = '/home/gi75qag/models/archive/BEST_checkpoint_blocks2D_logos_9_cap_per_img_pair_1_min_word_freq_TRANSFORMER_31_3.pth.tar'  # model checkpoint
checkpoint_name = "BEST_checkpoint_blocks2D_logos_9_cap_per_img_pair_1_min_word_freq_TRANSFORMER_31_3.pth.tar"
example_image_before_path = '/home/gi75qag/img_before_image_test_data_set_513.jpg'
example_image_after_path = '/home/gi75qag/img_after_image_test_data_set_513.jpg'
smooth = True
example_image_index = 513

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

# VISUALIZATION FUNCTION 
# IMPORTS
import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import ImageGrid
from itertools import count
import skimage.transform
import argparse
from skimage.transform import resize as imresize
from skimage.io import imread
from skimage.io import imsave
from PIL import Image
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn
#from nltk.translate.bleu_score import corpus_bleu
#from nltk.translate.bleu_score import SmoothingFunction

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}

# VISUALIZATION FUNCTION
def visualize_att(image_path, alphas, savingpath, smooth=True):
    """
    Visualizes caption with weights at every word.
    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    target_size = [14 * 24, 14 * 24] # [360, 480]

    image = Image.open(image_path)
    image = image.resize(target_size, Image.LANCZOS)

    plt.figure(figsize=(15, 10))
    plt.imshow(image)
    if smooth:
      alpha = skimage.transform.pyramid_expand(alphas.numpy(), upscale=24, sigma=8)
    else:
      alpha = skimage.transform.resize(alphas.numpy(), target_size)
    plt.imshow(alpha, alpha=0.8)
    plt.set_cmap(cm.Greys_r)
    plt.axis('off')
    plt.show()
    plt.savefig(savingpath, bbox_inches='tight', pad_inches=0)

# END OF VISUALIZATION FUNCTION


# EXECUTE EVALUATION ON TEST DATASET
print('USING CHECKPOINT: \n')
print(checkpoint_name)
print('\n')
print('VISUALIZING: ')

# 1. Visualize the 8 Attention Heads
loader = torch.utils.data.DataLoader(
    CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize]), img_return_mode='SPLITTED'),
    batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

hypotheses = list()
references = list()
before_image_alphas_tensor = ''
encoder_decoder_mh_att_weights = ''

# For each image
for i, (img_before, img_after, caps, caplens, allcaps) in enumerate(
          tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
      
  if i != example_image_index:
    continue
  if i > example_image_index:
    break

  # get best hypothesis from translation
  best_hypothesis, _, before_image_alphas, encoder_decoder_mh_att_weights = translate(img_before, img_after,
                                    beam_size=beam_size,
                                    length_norm_coefficient=0.6)
      
  before_image_alphas_tensor = before_image_alphas
  encoder_decoder_mh_att_weights = encoder_decoder_mh_att_weights
  
  # References
  img_caps = allcaps[0].tolist()
  img_captions = list(
            map(lambda c: [rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
  references.append(img_captions)

  # Hypotheses
  hypotheses.append([w for w in best_hypothesis if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
  assert len(references) == len(hypotheses)

print('hypothesis: ')
print(hypotheses)
print('\n')
print('references: ')
print(references)
print(before_image_alphas_tensor.shape)
#print(after_image_alphas_tensor.shape)
before_image_alphas_tensor = before_image_alphas_tensor[:,:,0,:,:] # go from # (1, 1, 196, 196, 8) to #(1, 1, h, w, nb_heads)

# check for transition width of 2 or 8 
if (before_image_alphas_tensor.size(-1) == 2):
    before_image_alphas_tensor = before_image_alphas_tensor.view(1,1,14,14,2)
else:
    before_image_alphas_tensor = before_image_alphas_tensor.view(1,1,14,14,8)

#after_image_alphas_tensor = after_image_alphas_tensor[:,:,0,:,:] # go from # (1,1, 196, 196, 8) to #(1, 1, h, w, nb_heads)
#after_image_alphas_tensor = after_image_alphas_tensor.view(1,1,14,14,8)
after_image_alphas_tensor = before_image_alphas_tensor # because of identical trajectory
print(before_image_alphas_tensor.shape) # [1, 1, 14, 14, 8] 
print(after_image_alphas_tensor.shape)
before_image_alphas_tensor = before_image_alphas_tensor.to('cpu')
after_image_alphas_tensor = after_image_alphas_tensor.to('cpu')
# => apply the function from duda 8 times here 

# set the ranges for the transition width cases of 2 and 8
if (before_image_alphas_tensor.size(-1) == 2):
    before_range = 1
    after_range = 2
else:
    before_range = 4
    after_range = 8

# Visualize attention on before image
for i in range(0,before_range):
  before_image_alphas_tensor = torch.squeeze(before_image_alphas_tensor) # [14,14,8]
  alpha_bef = torch.FloatTensor(before_image_alphas_tensor[:,:,i])
  alpha_bef = alpha_bef.detach() # get rid of gradients
  savingpath = '/home/gi75qag/' + 'before_image_' + str(i) + '.png'
  visualize_att(example_image_before_path, alpha_bef, savingpath)

# Visualize attention on after image
for i in range(before_range,after_range):
  after_image_alphas_tensor = torch.squeeze(after_image_alphas_tensor) # [14,14,8]
  alpha_aft = torch.FloatTensor(after_image_alphas_tensor[:,:,i])
  alpha_aft = alpha_aft.detach() # get rid of gradients
  savingpath = '/home/gi75qag/' + 'after_image_' + str(i) + '.png'
  visualize_att(example_image_after_path, alpha_aft, savingpath)
  
# Plot grid image of all attention weighted images
image_list = []
for i in range(0,after_range):
    if (i < before_range):
        image_path = '/home/gi75qag/' + 'before_image_' + str(i) + '.png'
    else:
        image_path = '/home/gi75qag/' + 'after_image_' + str(i) + '.png'
    image = Image.open(image_path)
    image_list.append(image)
    
fig = plt.figure(figsize=(30., 40.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, before_range),  # creates 2x2 grid of axes
                 axes_pad=0.9,  # pad between axes in inch.
                 )

    
for index, ax, im in zip(count(), grid, image_list):
    ax.imshow(im)
    ax.set_title(str(index), fontsize=50)
    ax.axis('off')
    
grid_image_path = '/home/gi75qag/grid_image.png'
plt.savefig(grid_image_path, bbox_inches='tight', pad_inches=0) 



