# IMPORTS
import torch
from torch import nn
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys
sys.path.append('.')


# NEAREST NEIGHBOR BASELINE MODEL

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


# NEAREST NEIGHBOR CLASSIFIER
class NearestNeighbor():
  """ Compute nearest neighbors for each query point.
  """
  def __init__(self, model_dataloader, image_encoder):
    self.model_dataloader = model_dataloader
    self.image_encoder = image_encoder

  def apply(self, test_image): 
    """ Return index, caption and distance to nearest neighbor of test_image [1,3,720,480]
    """
    # Move Image_Encoder to GPU, if possible and set model to evaluation mode for suppressing backpropagation
    self.image_encoder.fine_tune(fine_tune=False)
    self.image_encoder = self.image_encoder.to(device)
    self.image_encoder.eval()
    # Free GPU Memory 
    torch.cuda.empty_cache()
    with torch.no_grad():
      test_image = test_image.to(device)
      test_image_feature_vector = self.image_encoder(test_image)
      #Flatten the feature vector to [1, 401408]
      test_image_feature_vector = test_image_feature_vector.reshape(1, -1) 
      #print(test_image_feature_vector.shape)
      # Initialize measures
      nearest_neighbor_distance = float('inf')
      nearest_neighbor_index = 0
      nearest_neighbor_caption = ''
      nearest_neighbor_caplen = 0
      for j, (model_images, model_caps, model_caplens) in enumerate(self.model_dataloader):
        #print('model batch index: ', j)
        # Move to GPU device, if available
        model_images = model_images.to(device)
        model_images_batch_size = model_images.shape[0]
        # Extract Feature Vector from model image
        model_images_feature_vectors = self.image_encoder(model_images)
        # Flatten the feature vectors to [4, 401408]]
        model_images_feature_vectors = model_images_feature_vectors.reshape(model_images_batch_size, -1)
        #print(model_images_feature_vectors.shape)
        distance_tensor = torch.norm(model_images_feature_vectors - test_image_feature_vector, dim=1, p=None)
        nearest_neighbour = distance_tensor.topk(1, largest=False)
        # Get distance as scalar from tensor
        distance = nearest_neighbour.values[0]
        if distance <= nearest_neighbor_distance:
          nearest_neighbor_distance = distance
          # Use batch number to reconstruct image index
          nearest_neighbor_index = nearest_neighbour.indices[0] + j*4 
          # Get caption of nearest neighbor
          nearest_neighbor_caption = model_caps[nearest_neighbour.indices[0]]
          # Get caplen of nearest neighbor
          nearest_neighbor_caplen = model_caplens[nearest_neighbour.indices[0]]

    return nearest_neighbor_index, nearest_neighbor_caption, nearest_neighbor_caplen, nearest_neighbor_distance