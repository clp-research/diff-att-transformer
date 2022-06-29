# IMPORTS
import torch.backends.cudnn as cudnn
import torch.utils.data
from dotted_dict import DottedDict
import sys
import os
import json

from utils.transformer_utils import get_lr, get_positional_encoding

sys.path.append('.')
cudnn.benchmark = False  # since input tensor size is variable

# TRAIN TRANSFORMER (DIFF/CHANGE-ATTENTION IN IMAGE ENCODER) MODEL
config = DottedDict()

# Checkpoints
config.save_dir = "/home/users/sadler/cache/052_block_instruct_transformer/models/blocks"
config.model_name = "diff_att_8_MODEL_PRE"
config.transformer_checkpoint = None  # 'diff_att_8_MODEL_PRE.pth.tar'   # 'transformer_checkpoint.pth.tar'  # path to model checkpoint, None if none
config.start_epoch = 0  # start at this epoch (might be replaced by loaded checkpoint)

# Staged training
config.fine_tune_image_encoder = False  # switch in case of staged training
config.early_stopping = 20

# Experiment hyper-parameter
config.nb_single_image_attention_heads = 8  # self attention heads on multi headed image attention for embedding the images

# Model parameters
config.d_model = 512  # size of vectors throughout the transformer model
config.n_heads = 8  # number of heads in the multi-head attention
config.d_queries = 64  # size of query vectors (and also the size of the key vectors) in the multi-head attention
config.d_values = 64  # size of value vectors in the multi-head attention
config.d_inner = 2048  # an intermediate size in the position-wise FC
config.n_layers = 6  # number of layers in the Encoder and Decoder
config.dropout = 0.1  # dropout probability
config.positional_encoding = get_positional_encoding(d_model=config.d_model,
                                                     max_length=102)  # positional encodings up to the maximum possible pad-length
config.encoder_dim = 2048  # dimension of the image encoder || channels of the image encoder
config.encoded_image_size = 14  # pixels of width|height of an encoded image => num_pixels = 196
config.pad_length = 102  # pad length of the sequences, defined by maximum length of a caption during generation of the dataset
config.attention_dim = 128

# Learning parameters
config.average_tokens_per_single_batch = 15
config.batch_size = 8
config.tokens_in_batch = config.batch_size * config.average_tokens_per_single_batch  # 2000 in vanilla transformer  # batch size in target language tokens
config.batches_per_step = 1  # original: 25000 // tokens_in_batch  # perform a training step, i.e. update parameters, once every so many batches
config.print_frequency = 20  # print status once every so many steps
config.n_steps = 10000  # 100000  # number of training steps
config.warmup_steps = 80  # 8000  # number of warmup steps where learning rate is increased linearly; twice the value in the paper, as in the official transformer repo.
config.step = 1  # the step number, start from 1 to prevent math error in the next line
config.lr = get_lr(step=config.step, d_model=config.d_model,
                   warmup_steps=config.warmup_steps)  # see utils.py for learning rate schedule; twice the schedule in the paper, as in the official transformer repo.
config.image_encoder_lr = 1e-4
config.lambda_l1 = 2.5e-3  # regularization parameter for attention map activations
config.transformer_lr = 1e-4
config.betas = (0.9, 0.98)  # beta coefficients in the Adam optimizer
config.epsilon = 1e-9  # epsilon term in the Adam optimizer
config.label_smoothing = 0.1  # label smoothing co-efficient in the Cross Entropy loss
config.workers = 1

# State parameters (are updated during training)
config.best_bleu4 = 0.
config.epochs_since_improvement = 0

# Data parameters
config.data_folder = '/home/users/sadler/data/blockworld_pre'  # folder with data files saved by create_input_files.py
config.data_name = 'blocks2D_logos_9_cap_per_img_pair_1_min_word_freq'  # base name shared by data files

# Read word map
word_map_file = os.path.join(config.data_folder, 'WORDMAP_' + config.data_name + '.json')
with open(word_map_file, 'r') as j:
    config.word_map = json.load(j)
config.vocab_size = len(config.word_map)

from training.train_transformer import perform_training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CPU isn't really practical here
perform_training(config, device)
