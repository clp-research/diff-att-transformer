# IMPORTS
import json
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.distributions import Categorical
from nltk.translate.bleu_score import corpus_bleu 
from nltk.translate.bleu_score import SmoothingFunction
import sys
sys.path.append('.')
from models.duda_model import *
from utils.duda_utils import *
from utils.dataset import *


# TRAIN DUDA MODEL

# Data parameters
data_folder = '/home/gi75qag/dataset/preprocessed_dataset'  # folder with data files saved by create_input_files.py
data_name = 'blocks2D_logos_9_cap_per_img_pair_1_min_word_freq'  # base name shared by data files

# Model parameters 
embed_dim = 512
decoder_dim = 512
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
captions_per_image = 9
feature_dim = 2048
attention_dim = 128  # width of the attention window in the alpha image
hidden_dim = 512 # dimension of hidden layer of lstm for next word classification in dynamic speaker

# Training parameters
start_epoch = 0
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0
batch_size = 8
workers = 1
decoder_lr = 1e-4
encoder_lr = 1e-4
image_encoder_lr = 1e-4
grad_clip = 5.
alpha_c = 1.
lambda_l1 = 2.5e-3
lambda_ent = 1e-4
best_bleu4 = 0.
print_freq = 100 
checkpoint = None #'/home/gi75qag/models/BEST_checkpoint_blocks2D_logos_9_cap_per_img_pair_1_min_word_freq_DUDA_NOREG_NODOM.pth.tar'  
fine_tune_image_encoder = False

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def get_key(dict_, value):
  return [k for k, v in dict_.items() if v == value]
  


def build_duda_model():
  global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, data_name, fine_tune_image_encoder


  # Read word map
  word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
  with open(word_map_file, 'r') as j:
    word_map = json.load(j)

  # Initialize / load from checkpoint
  if checkpoint is None:
    image_feature_encoder = Encoder()
    image_feature_encoder.fine_tune(fine_tune_image_encoder)
    encoder = DualAttention(attention_dim=attention_dim,
                          feature_dim = feature_dim)
    decoder = DynamicSpeaker(feature_dim = feature_dim,
                           embed_dim = embed_dim,
                           vocab_size = len(word_map),
                           hidden_dim = hidden_dim,
                           dropout=dropout)
    image_encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, image_feature_encoder.parameters()),
                                             lr=image_encoder_lr) if fine_tune_image_encoder else None  
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                       lr=encoder_lr)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                       lr=decoder_lr)
  else:
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    epochs_since_improvement = checkpoint['epochs_since_improvement']
    best_bleu4 = checkpoint['bleu-4']
    decoder = checkpoint['decoder']
    decoder_optimizer = checkpoint['decoder_optimizer']
    encoder = checkpoint['encoder']
    encoder_optimizer = checkpoint['encoder_optimizer']
    image_feature_encoder = checkpoint['image_encoder']
    image_encoder_optimizer = checkpoint['image_encoder_optimizer']
    if fine_tune_image_encoder is True and image_encoder_optimizer is None:
      image_feature_encoder.fine_tune(fine_tune_image_encoder)
      image_encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, image_feature_encoder.parameters()),
                                                 lr=image_encoder_lr)


  # Move to GPU, if available
  decoder = decoder.to(device)
  encoder = encoder.to(device)
  image_feature_encoder = image_feature_encoder.to(device)

  # Loss function
  criterion = nn.CrossEntropyLoss().to(device)

  # Custom data loaders
  train_loader = torch.utils.data.DataLoader(
    CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize]), img_return_mode='SPLITTED'),
    batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

  val_loader = torch.utils.data.DataLoader(
    CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize]), img_return_mode='SPLITTED'),
    batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
  
  # Epochs
  for epoch in range(start_epoch, epochs):
    print("epoch : " + str(epoch))

    # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 15
    if epochs_since_improvement == 40:
      break
      if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
        adjust_learning_rate(decoder_optimizer, 0.8)
        adjust_learning_rate(encoder_optimizer, 0.8)
        if fine_tune_image_encoder:
          adjust_learning_rate(image_encoder_optimizer, 0.8)

    train(train_loader=train_loader,
          image_feature_encoder=image_feature_encoder,
          encoder=encoder,
          decoder=decoder,
          criterion=criterion,
          image_encoder_optimizer=image_encoder_optimizer,
          encoder_optimizer=encoder_optimizer,
          decoder_optimizer=decoder_optimizer,
          epoch=epoch,
          word_map=word_map
          )

    # One epoch's validation
    recent_bleu4 = validate(val_loader=val_loader,
                            image_feature_encoder=image_feature_encoder,
                            encoder=encoder,
                            decoder=decoder,
                            criterion=criterion,
                            word_map=word_map)

    # check if there was an improvement
    is_best = recent_bleu4 > best_bleu4
    best_bleu4 = max(recent_bleu4, best_bleu4)
    if not is_best:
      epochs_since_improvement += 1
      print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
    else:
      epochs_since_improvement = 0

    # Save checkpoint    
    save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer, recent_bleu4, is_best, model_name='DUDA', image_encoder_optimizer=image_encoder_optimizer, image_encoder=image_feature_encoder)

def train(train_loader, image_feature_encoder, encoder, decoder, criterion, image_encoder_optimizer, encoder_optimizer, decoder_optimizer, epoch, word_map):
  image_feature_encoder.eval() # train() 
  encoder.train()
  decoder.train()

  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  top3accs = AverageMeter()
 
  start = time.time()
  
  # Batches
  for i, (imgs1, imgs2, caps, caplens) in enumerate(train_loader):
    data_time.update(time.time() - start)

    # Move to GPU, if available
    imgs1 = imgs1.to(device)
    imgs2 = imgs2.to(device)
    caps = caps.to(device)
    caplens = caplens.to(device)

    # Extract Image Features/Image Encoder
    imgs1 = image_feature_encoder(imgs1)
    imgs2 = image_feature_encoder(imgs2)

    # Forward prop.
    l_bef, l_aft, alpha_bef, alpha_aft = encoder(imgs1, imgs2)
    scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(l_bef, l_aft, caps, caplens)

    targets = caps_sorted[:, 1:]

    scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
    targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

    loss = criterion(scores, targets)

    # Add Regularization
    # Add doubly stochastic attention regularization -> which alphas to penalize? alphas on before and after images + alphas on concatenated attended image?
    #loss += alpha_c * (((1. - alphas.sum(dim=1)) ** 2).mean() + ((1. - alpha_bef.sum(dim=1)) ** 2).mean() + ((1. - alpha_aft.sum(dim=1)) ** 2).mean())/3 
    # 1. add L1 regularization over attention masks generated by dynamic attention modules
    #batch_size = alpha_bef.size(0)
    #alpha_bef = alpha_bef.view(batch_size, -1) # flatten the attention mask
    #alpha_aft = alpha_aft.view(batch_size, -1) # flatten the attention mask
    #l1_regularization_alpha_masks = lambda_l1 * (alpha_bef.sum(dim=1).mean() + alpha_aft.sum(dim=1).mean()) # compute the mean of the attention activation over the whole batch
    # 2. add entropy regularization over attention weights generated by dynamic speaker 
    #alphas = alphas.mean(dim=1) # sum up over decode lengths
    #alphas = alphas.mean(dim=1) # sum up over averaged attentions on the 3 images
    #alphas_entropy = Categorical(probs = alphas).entropy()
    #l_alpha_entropy = lambda_ent * alphas_entropy
    # add regularization terms to overall loss like described in paper
    #loss = loss + l1_regularization_alpha_masks - l_alpha_entropy
    
    # Back prop.
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    if image_encoder_optimizer is not None:
        image_encoder_optimizer.zero_grad()
    loss.backward()

    # Clip gradients
    #if grad_clip is not None:
    #  clip_gradient(decoder_optimizer, grad_clip)
    #  clip_gradient(encoder_optimizer, grad_clip)
    #  if image_encoder_optimizer is not None:
    #    clip_gradient(image_encoder_optimizer, grad_clip)

    # Update weights
    encoder_optimizer.step()
    decoder_optimizer.step()
    if image_encoder_optimizer is not None:
        image_encoder_optimizer.step()

    # Keep track of metrics
    top3 = accuracy(scores, targets, 3)
    losses.update(loss.item(), sum(decode_lengths))
    top3accs.update(top3, sum(decode_lengths))
    batch_time.update(time.time() - start)

    start = time.time()
    
    # Print status
    if i % print_freq == 0:
      print('Epoch: [{0}][{1}/{2}]\t'
            'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Top-3 Accuracy {top3.val:.3f} ({top3.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                    batch_time=batch_time,
                                                                    data_time=data_time,
                                                                    loss=losses,
                                                                    top3=top3accs))
  
def validate(val_loader, image_feature_encoder, encoder, decoder, criterion, word_map):
  image_feature_encoder.eval()
  encoder.eval()
  decoder.eval()

  batch_time = AverageMeter()
  losses = AverageMeter()
  top3accs = AverageMeter()

  start = time.time()

  references = list()
  hypotheses = list()

  with torch.no_grad():
    # Batches
    for i, (imgs1, imgs2, caps, caplens, allcaps) in enumerate(val_loader):
      imgs1 = imgs1.to(device)
      imgs2 = imgs2.to(device)
      caps = caps.to(device)
      caplens = caplens.to(device)

      # Extract Image Features/Image Encoder
      imgs1 = image_feature_encoder(imgs1)
      imgs2 = image_feature_encoder(imgs2)

      # Forward prop.
      l_bef, l_aft, alpha_bef, alpha_aft = encoder(imgs1, imgs2)
      scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(l_bef, l_aft, caps, caplens)

      targets = caps_sorted[:, 1:]

      scores_copy = scores.clone()
      scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
      targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

      loss = criterion(scores, targets)

      losses.update(loss.item(), sum(decode_lengths))
      top3 = accuracy(scores, targets, 3)
      top3accs.update(top3, sum(decode_lengths))
      batch_time.update(time.time() - start)

      start = time.time()

      if i % print_freq == 0:
        print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-3 Accuracy {top3.val:.3f} ({top3.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top3=top3accs))

      # References
      allcaps = allcaps[sort_ind]
      for j in range(allcaps.shape[0]):
        img_caps = allcaps[j].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}], img_caps))
        references.append(img_captions)

      # Hypotheses
      _, preds = torch.max(scores_copy, dim=2)
      preds = preds.tolist()
      temp_preds = list()
      for j, p in enumerate(preds):
        temp_preds.append(preds[j][:decode_lengths[j]])
      preds = temp_preds
      hypotheses.extend(preds)
      assert len(references) == len(hypotheses)
    
    # Calculate BLEU-4 Score 
    # Overcome missing overlaps with smoothing, described here: https://www.nltk.org/_modules/nltk/translate/bleu_score.html
    #smoothing_fn = SmoothingFunction().method4
    #bleu4 = corpus_bleu(references, hypotheses, smoothing_function=smoothing_fn)
    bleu4 = corpus_bleu(references, hypotheses)
    #print(hypotheses)
    print('\n * LOSS - {loss.avg:.3f}, TOP-3 ACCURACY - {top3.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top3=top3accs,
                bleu=bleu4))

    return bleu4
# END OF TRAINING

# EXECUTION
build_duda_model()
# END OF EXECUTION