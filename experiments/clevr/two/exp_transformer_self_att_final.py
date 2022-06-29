import torch.backends.cudnn as cudnn
import torch.utils.data
from dotted_dict import DottedDict
import sys
import os
import json

from training.train_transformer import perform_training
from utils.tensorboard import get_summary_writer_for
from utils.transformer_utils import get_lr, get_positional_encoding

sys.path.append('.')
cudnn.benchmark = False  # since input tensor size is variable

# TRAIN TRANSFORMER (DIFF/CHANGE-ATTENTION IN IMAGE ENCODER) MODEL
config = DottedDict()

sc_only = "_sc"  # or change to empty string

# Checkpoints
config.save_dir = "/home/users/sadler/cache/052_block_instruct_transformer/models/clevr_change" + sc_only
config.model_name = "self_att_2_MODEL_FINAL"
config.transformer_checkpoint = None  # if None train from scratch else restart the training
config.start_epoch = 0  # start at this epoch (might be replaced by loaded checkpoint)
config.max_epochs = 200

# Staged training
config.fine_tune_image_encoder = False  # for CLEVR we fix the image encoder
config.batch_size = 16  # increase batch size for fixed encoder (more RAM available)
config.early_stopping = 5
config.workers = 0
config.dry_run = False
print("Dry Run: " + str(config.dry_run))

# Experiment hyper-parameter
config.nb_single_image_attention_heads = 2  # self attention heads on multi headed image attention for embedding the images

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
config.tokens_in_batch = config.batch_size * config.average_tokens_per_single_batch  # 2000 in vanilla transformer  # batch size in target language tokens
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

# State parameters (are updated during training)
config.best_bleu4 = 0.
config.epochs_since_improvement = 0

# Data parameters
config.data_folder = "/home/users/sadler/data/clevr_change_pre" + sc_only  # folder with data files saved by create_input_files.py
config.data_name = 'CLEVR_CHANGE_9_cap_per_img_pair_1_min_word_freq'  # base name shared by data files

# Read word map
word_map_file = os.path.join(config.data_folder, 'WORDMAP_' + config.data_name + '.json')
with open(word_map_file, 'r') as j:
    config.word_map = json.load(j)
config.vocab_size = len(config.word_map)

from models.transformer_self_att_model import Transformer, ImageEncoder

# Initialize model or load checkpoint
if config.transformer_checkpoint is None:
    model = Transformer(vocab_size=config.vocab_size,
                        positional_encoding=config.positional_encoding,
                        nb_single_image_attention_heads=config.nb_single_image_attention_heads,
                        pad_length=config.pad_length,
                        attention_dim=config.attention_dim,
                        d_model=config.d_model,
                        n_heads=config.n_heads,
                        d_queries=config.d_queries,
                        d_values=config.d_values,
                        d_inner=config.d_inner,
                        n_layers=config.n_layers,
                        dropout=config.dropout)
    optimizer = torch.optim.Adam(params=[p for p in model.parameters() if p.requires_grad],
                                 lr=config.transformer_lr,
                                 betas=config.betas,
                                 eps=config.epsilon)
    image_feature_encoder = ImageEncoder()
    image_feature_encoder.fine_tune(config.fine_tune_image_encoder)
    image_encoder_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, image_feature_encoder.parameters()),
        lr=config.image_encoder_lr) if config.fine_tune_image_encoder else None

else:
    checkpoint = torch.load(config.transformer_checkpoint)
    config.start_epoch = checkpoint['epoch'] + 1  # overwrite start epoch
    print('\nLoaded checkpoint from epoch %d.\n' % config.start_epoch)
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    image_feature_encoder = checkpoint['image_encoder']
    image_encoder_optimizer = checkpoint['image_encoder_optimizer']
    if config.fine_tune_image_encoder is True and image_encoder_optimizer is None:
        image_feature_encoder.fine_tune(config.fine_tune_image_encoder)
        image_encoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, image_feature_encoder.parameters()),
            lr=config.image_encoder_lr)

models = {"transformer": model, "resnet": image_feature_encoder}
optimizers = {"transformer": optimizer, "resnet": image_encoder_optimizer}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CPU isn't really practical here
tblogger = get_summary_writer_for(config.model_name)
perform_training(config, models, optimizers, device, tblogger)
tblogger.close()
