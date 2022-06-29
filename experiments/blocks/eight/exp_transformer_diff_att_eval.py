import json
import os

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from dotted_dict import DottedDict

import sys

from utils.dataset import CaptionDataset

sys.path.append('.')

from evaluation.eval_transformer_diff_att_blocks import perform_eval

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
"""
target_detection_ratio:  0.7323511356660528
landmark_detection_ratio:  0.6780233271945979
spatial_description_detection_ratio:  0.8661755678330264
The number of references is 1629
The number of hypotheses is 1629
{'testlen': 17888, 'reflen': 18080, 'guess': [17888, 16259, 14630, 13001], 'correct': [15878, 11523, 8437, 6234]}
ratio: 0.9893805309733966
{
'Bleu_1': 0.8781577372825368, 
'Bleu_2': 0.7846778471558987, 
'Bleu_3': 0.7055942112782194, 
'Bleu_4': 0.6389228158114025, 
'METEOR': 0.4103024021825825, 
'ROUGE_L': 0.7751089942535228, 
'CIDEr': 1.3957583367034623
}
"""
config = DottedDict()

# model parameters
# config.captions_per_image = 9 (never used)

config.beam_size = 5
config.nb_heads = 8  # number of attention heads on IMAGE used in the model -> important for figuring out visual word/sentence size

# Data parameters
data_folder = '/home/users/sadler/data/blockworld_pre'  # folder with data files saved by create_input_files.py
data_name = 'blocks2D_logos_9_cap_per_img_pair_1_min_word_freq'  # base name shared by data files
data_split = "TEST"

# Load word map (word2ix)
word_map_file = os.path.join(config.data_folder, 'WORDMAP_' + config.data_name + '.json')
with open(word_map_file, 'r') as j:
    config.word_map = json.load(j)
config.rev_word_map = {v: k for k, v in config.word_map.items()}
config.vocab_size = len(config.word_map)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CPU isn't really practical here

# Load model
config.checkpoint_name = "diff_att_8_MODEL_FINAL"

save_dir = "/home/users/sadler/cache/052_block_instruct_transformer/models/blocks"
transformer_checkpoint = save_dir + "/BEST_{}.pth.tar".format(config.checkpoint_name)
checkpoint = torch.load(transformer_checkpoint, map_location=torch.device('cuda'))

model = checkpoint['model'].to(device)
model.eval()
image_feature_encoder = checkpoint['image_encoder'].to(device)
image_feature_encoder.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
dataset = CaptionDataset(data_folder, data_name, data_split,
                         transform=transforms.Compose([normalize]),
                         img_return_mode='SPLITTED')
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=1,
                                          pin_memory=True)
perform_eval(config, data_loader, model, image_feature_encoder, device)
