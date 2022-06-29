# IMPORTS
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from torch import nn
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys
sys.path.append('.')


# DUDA MODEL

# ENCODER
class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        #resnet = torchvision.models.resnet101()
        #resnet.load_state_dict(torch.load("/home/gi75qag/resnet101.pth"))
        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101
        #resnet = torch.load("/home/gi75qag/resnet101.pth")

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
# END OF ENCODER

# DUAL ATTENTION NETWORK
class DualAttention(nn.Module):
  """
  Dual attention network.
  """

  def __init__(self, attention_dim, feature_dim):
    """
    """
    super(DualAttention, self).__init__()
    self.conv1 = nn.Conv2d(feature_dim*2, attention_dim, kernel_size=1, padding=0)
    self.relu = nn.ReLU()
    self.conv2 = nn.Conv2d(attention_dim, 1, kernel_size=1, padding=0)
    self.sigmoid = nn.Sigmoid()

  def forward(self, img_feat1, img_feat2):
    # Permute Input Image from Encoder to fit to dual attention mechanism
    # img_feat1.shape | img_feat2.shape => (batch_size, encoded_image_size, encoded_image_size, 2048)
    img_feat1 = img_feat1.permute(0, 3, 1, 2)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
    img_feat2 = img_feat2.permute(0, 3, 1, 2)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
    # img_feat1 (batch_size, feature_dim, h, w)
    batch_size = img_feat1.size(0)
    feature_dim = img_feat1.size(1)
    
    img_diff = img_feat2 - img_feat1

    img_feat1_d = torch.cat([img_feat1, img_diff], dim=1)
    img_feat2_d = torch.cat([img_feat2, img_diff], dim=1)

    img_feat1_d = self.conv1(img_feat1_d)
    img_feat2_d = self.conv1(img_feat2_d)

    img_feat1_d = self.relu(img_feat1_d)
    img_feat2_d = self.relu(img_feat2_d)

    img_feat1_d = self.conv2(img_feat1_d)
    img_feat2_d = self.conv2(img_feat2_d)
    
    # To this point
    # img_feat1, img_feat2 have dimension
    # (batch_size, hidden_dim, h, w)

    alpha_img1 = self.sigmoid(img_feat1_d)
    alpha_img2 = self.sigmoid(img_feat2_d)

    # To this point
    # alpha_img1, alpha_img2 have dimension
    # (batch_size, 1, h, w)

    img_feat1 = img_feat1*(alpha_img1.repeat(1,2048,1,1))
    img_feat2 = img_feat2*(alpha_img2.repeat(1,2048,1,1))
    
    # (batch_size,feature_dim,h,w) 

    img_feat1 = img_feat1.sum(-2).sum(-1).view(batch_size, -1) # (batch_size,feature_dim)
    img_feat2 = img_feat2.sum(-2).sum(-1).view(batch_size, -1)
    #img_feat1 = img_feat1.view(batch_size, -1)
    #img_feat2 = img_feat2.view(batch_size, -1)

    return img_feat1, img_feat2, alpha_img1, alpha_img2
# END OF DUAL ATTENTION NETWORK

# DYNAMIC SPEAKER NETWORK
class DynamicSpeaker(nn.Module):
  """
  Dynamic speaker network.
  """

  def __init__(self, feature_dim, embed_dim, vocab_size, hidden_dim, dropout):
    """
    """
    super(DynamicSpeaker, self).__init__()

    self.feature_dim = feature_dim
    self.embed_dim = embed_dim
    self.hidden_dim = hidden_dim
    self.vocab_size = vocab_size
    self.dropout = dropout
    self.softmax = nn.Softmax(dim=1) ##### TODO ##### CHECK

    # embedding layer
    self.embedding = nn.Embedding(vocab_size, embed_dim)
    self.dropout = nn.Dropout(p=self.dropout)

    # dynamic temporal attention LSTM 
    self.dynamic_att = nn.LSTMCell(hidden_dim*2,hidden_dim, bias=True)
    self.init_hda = nn.Linear(hidden_dim, hidden_dim)  # linear layer to find initial hidden state of LSTMCell
    self.init_cda = nn.Linear(hidden_dim, hidden_dim)  # linear layer to find initial cell state of LSTMCell

    # dynamic speaker/decoder LSTM
    self.decode_step = nn.LSTMCell(embed_dim + feature_dim, hidden_dim, bias=True)
    self.init_hds = nn.Linear(feature_dim, hidden_dim)  # linear layer to find initial hidden state of LSTMCell
    self.init_cds = nn.Linear(feature_dim, hidden_dim)  # linear layer to find initial cell state of LSTMCell

    # learnable weight matrices
    self.relu = nn.ReLU()
    self.wd1 = nn.Linear(feature_dim*3, hidden_dim)
    self.wd2 = nn.Linear(hidden_dim, 3) ##### TODO ##### CHECK
    # Linear layer to find scores over vocabulary
    self.wdc = nn.Linear(hidden_dim, vocab_size)
    self.init_weights() # initialize some layers with the uniform distribution

  def init_weights(self):
    """
    Initializes some parameters with values from the uniform distribution, for easier convergence
    """
    self.embedding.weight.data.uniform_(-0.1,0.1)
    self.wd1.bias.data.fill_(0)
    self.wd1.weight.data.uniform_(-0.1,0.1)
    self.wd2.bias.data.fill_(0)
    self.wd2.weight.data.uniform_(-0.1,0.1)
    self.wdc.bias.data.fill_(0)
    self.wdc.weight.data.uniform_(-0.1,0.1)
    #for name, param in self.dynamic_att.named_parameters():
    #    if 'bias' in name:
    #        nn.init.constant(param, 0.0)
    #    elif 'weight' in name:
    #        nn.init.xavier_normal(param)   
    #for name, param in self.decode_step.named_parameters():
    #    if 'bias' in name:
    #        nn.init.constant(param, 0.0)
    #    elif 'weight' in name:
    #        nn.init.xavier_normal(param)        

  def init_da_hidden_state(self, l_total):
        """
        Creates the initial hidden and cell states for the decoder's dynamic attention LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        #mean_l_total = l_total.mean(dim=0)
        h_da = self.init_hda(l_total)  # (batch_size, hidden_dim)
        c_da = self.init_cda(l_total)
        return h_da, c_da 
    
  def init_ds_hidden_state(self, l_dyn):
        """
        Creates the initial hidden and cell states for the decoder's dynamic speaker LSTM based on the stacked attended latent images. 
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        #mean_l_dyn = l_dyn.mean(dim=0)
        h_ds = self.init_hds(l_dyn)  # (batch_size, hidden_dim)
        c_ds = self.init_cds(l_dyn)
        return h_ds, c_ds 

  def forward(self, l_bef, l_aft, encoded_captions, caption_lengths):
    # To this point,
    # l_bef, l_aft have dimension
    # (batch_size, feature_dim)
  
    batch_size = l_bef.size(0)
    feature_dim = l_bef.size(1)

    l_diff = torch.sub(l_aft,l_bef)
    # Stack attention weighted encoded before and after images and difference image on top of each other
    l_total = torch.cat([l_bef,l_aft,l_diff],dim=1)
    # apply classic MLP attention on stacked image
    l_total = self.relu(self.wd1(l_total)) # (batch_size, hidden_dim)
    # initialize weighted dynamic attention of before, after and difference image by their mean
    l_dyn = torch.add(l_bef, l_aft)
    l_dyn = torch.add(l_dyn, l_diff)
    l_dyn = torch.div(l_dyn, 3)

    # Sort input data by decreasing lengths
    caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
    l_diff = l_diff[sort_ind]
    l_total = l_total[sort_ind]
    l_bef = l_bef[sort_ind]
    l_aft = l_aft[sort_ind]
    l_dyn = l_dyn[sort_ind]
    encoded_captions = encoded_captions[sort_ind]
    
    # initialize LSTM hidden state and cell state of dynamic temporal attention RNN
    #h_da, c_da = self.init_da_hidden_state(l_total)  # (batch_size, hidden_dim)
    #h_da = h_da.to(device)
    #c_da = c_da.to(device)
    h_da = torch.zeros(batch_size, self.hidden_dim).to(device)  ## TODO ## 
    c_da = torch.zeros(batch_size, self.hidden_dim).to(device)

    # initialize LSTM hidden state and cell state of dynamic speaker RNN
    #h_ds, c_ds = self.init_ds_hidden_state(l_dyn) # (batch_size, hidden_dim)
    #h_ds = h_ds.to(device)
    #c_ds = c_ds.to(device)
    h_ds = torch.zeros(batch_size, self.hidden_dim).to(device)
    c_ds = torch.zeros(batch_size, self.hidden_dim).to(device)

    # Embedding
    embeddings = self.embedding(encoded_captions) # (batch_size, max_caption_length, embed_dim)

    decode_lengths = (caption_lengths - 1).tolist()

    predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(device)
    alphas = torch.zeros(batch_size, max(decode_lengths), 3).to(device) #TODO  ## is three ok?

    for t in range(max(decode_lengths)):
      batch_size_t = sum([l > t for l in decode_lengths])

      u_t = torch.cat([l_total[:batch_size_t], h_ds[:batch_size_t]],dim=1)
      h_da, c_da = self.dynamic_att(u_t[:batch_size_t], (h_da[:batch_size_t], c_da[:batch_size_t]))

      a_t = self.softmax(self.wd2(h_da)) #### (batch_size, 3)
     
      l_dyn = a_t[:,0].unsqueeze(1)*l_bef[:batch_size_t] + a_t[:,1].unsqueeze(1)*l_aft[:batch_size_t] + a_t[:,2].unsqueeze(1)*l_diff[:batch_size_t] 
      
      c_t = torch.cat([embeddings[:batch_size_t,t,:],l_dyn[:batch_size_t]], dim=1)
      
      h_ds, c_ds = self.decode_step(c_t, (h_ds[:batch_size_t], c_ds[:batch_size_t]))

      preds = self.wdc(h_ds)  # 
      predictions[:batch_size_t, t, :] = preds
      alphas[:batch_size_t,t,:] = a_t

    return predictions, encoded_captions, decode_lengths, alphas, sort_ind
# END OF DYNAMIC SPEAKER NETWORK
# END OF DUDA MODEL