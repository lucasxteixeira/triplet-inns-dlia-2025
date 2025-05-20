# Metric dissimilarity utils

import os
import random
import math
import pickle

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
import seaborn as sns
import umap

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering


#########################################################################################################
#                    Generate patches from an image with optional minimum patch count                   #
#########################################################################################################
def gen_patches(img, patch_size, min_patches=None, regular=True):
  """
  Generates patches from an input image with an optional minimum patch count and patch distribution.

  Parameters
  ----------
  img : numpy.ndarray
      Input image as a NumPy array of shape (height, width, channels).
  patch_size : tuple
      A tuple (patch_height, patch_width) specifying the size of each patch.
  min_patches : int, optional
      The minimum number of patches required. Defaults to None.
  regular : bool, optional
      If True, generates a regular grid of patches. If False, randomly drops some patches to match `min_patches`. Defaults to True.

  Returns
  -------
  numpy.ndarray
      A 4D NumPy array of shape (number_of_patches, patch_height, patch_width, channels) containing the generated patches.
  """

  # Gets the shape of the input image.
  input_shape = img.shape

  # Calculates the minimum number of rows and columns of patches to cover the image.
  n_rows = math.ceil(input_shape[0] / patch_size[0])
  n_cols = math.ceil(input_shape[1] / patch_size[1])

  # Total number of patches.
  n_patches = n_rows * n_cols

  # Adjusts the number of rows and columns to ensure at least 'min_patches' patches are created.
  if min_patches is not None:
    while min_patches > n_patches:
      row_ratio = input_shape[0] / n_rows / patch_size[0]
      col_ratio = input_shape[1] / n_cols / patch_size[1]
      if row_ratio > col_ratio:
        n_rows += 1
      else:
        n_cols += 1
      n_patches = n_rows * n_cols

  # Calculates overlap between patches.
  row_overlap = math.ceil(((patch_size[0] * n_rows) - input_shape[0]) / (n_rows - 1))
  col_overlap = math.ceil(((patch_size[1] * n_cols) - input_shape[1]) / (n_cols - 1))

  # Generate all starting pixels, except the last one.
  row_patches = np.arange(0, input_shape[0], patch_size[0] - row_overlap)[0:(n_rows - 1)]
  col_patches = np.arange(0, input_shape[1], patch_size[1] - col_overlap)[0:(n_cols - 1)]

  # Create the last starting pixel manually to avoid going larger than the input image.
  row_patches = np.append(row_patches, input_shape[0] - patch_size[0])
  col_patches = np.append(col_patches, input_shape[1] - patch_size[1])

  # Generate rows and cols patches.
  row_patches = [(i, i + patch_size[0]) for i in row_patches]
  col_patches = [(i, i + patch_size[1]) for i in col_patches]

  # Combine them
  patches_indices = [(i, j) for i in row_patches for j in col_patches]

  # If not regular, then drop some patches to match min_patches
  if not regular:
    n_drop = n_patches - min_patches
    if n_drop > 0:
      # Generate random indices to delete
      drop_indices = random.sample(range(n_patches), n_drop)
      # Create a new list without the selected elements
      patches_indices = [patches_indices[i] for i in range(n_patches) if i not in drop_indices]
      # Update the number of patches
      n_patches = min_patches

  patches = np.zeros((n_patches, patch_size[0], patch_size[1], input_shape[2]), dtype=np.float32)

  # Extract patches from the image based on calculated indices.
  for patch_i in range(n_patches):
    row, col = patches_indices[patch_i]
    patches[patch_i] = img[row[0]:row[1], col[0]:col[1], :]

  # Normalize the patches if the image data type is 'uint8'.
  if img.dtype == "uint8":
    patches = (patches / 255).astype(np.float32)

  return patches


#########################################################################################################
#                       Generate a batch of image for training purposes                                 #
#########################################################################################################
def pair_batch(batch_size, X, Y, encoded=False, augment=False, size=None, device=None):
  """
  Generates a batch of pairs (images or encoded samples) and their corresponding classes for training purposes.

  Parameters
  ----------
  batch_size : int
      The number of pairs to generate.
  X : numpy.ndarray
      The input data (either image dataset or already encoded samples).
      If `encoded` is False, X is a NumPy array containing images of shape (num_samples, height, width, channels).
  Y : numpy.ndarray
      The class labels for the input images as a NumPy array of shape (num_samples,).
  encoded : bool, optional
      If True, the input `X` is already encoded and augmentation is skipped. Defaults to False.
  augment : bool, optional
      If True, applies data augmentation to the images. Defaults to False.
  size : tuple, optional
      The desired image size to resize to. Only applicable if `X` contains image data. Defaults to None.
  device : torch.device, optional
      The device to move the output tensors to (e.g., `torch.device('cuda')` for GPU). Defaults to None.

  Returns
  -------
  list of torch.Tensor
      A list containing two tensors of shape (batch_size, height, width, channels) for the image pairs,
      and a tensor of shape (batch_size,) for the corresponding class labels.
  """
  
  # Randomly select batch_size number of classes
  classes = np.random.choice(np.unique(Y), size=batch_size, replace=True)

  # Define the output shape based on the batch size, image size and encoding status
  if encoded:
    output_shape = (batch_size, X.shape[1])
  else:
    n_channels = 3 if len(X.shape) == 4 else 1
    if size is None:
      output_shape = (batch_size, n_channels, X.shape[1], X.shape[2])
    else:
      output_shape = (batch_size, n_channels, size[0], size[1])

  # Initialize arrays to store the pairs and their classes
  pairs = [torch.zeros(output_shape, dtype=torch.float32) for _ in range(2)]
  pairs.append(torch.from_numpy(classes))

  if not encoded:
    # Define the augmentation pipeline
    transform = A.Compose([
      A.RandomCrop(output_shape[2], output_shape[3]),
      A.VerticalFlip(),
      A.HorizontalFlip(),
      A.Rotate(),
      A.GaussianBlur(),
      A.RandomBrightnessContrast(),
      ToTensorV2()
    ])

  for i in range(batch_size):
    # Get indices of all samples that belong to the chosen class
    choices = np.where(Y == classes[i])[0]

    # Randomly select two samples of the same class
    idx_A = np.random.choice(choices)
    idx_B = np.random.choice(choices)

    if not encoded:
      img_A = transform(image=X[idx_A])["image"]
      img_B = transform(image=X[idx_B])["image"]

      # Save the samples to the pair list
      pairs[0][i] = img_A / 255.
      pairs[1][i] = img_B / 255.

    else:
      pairs[0][i] = torch.tensor(X[idx_A])
      pairs[1][i] = torch.tensor(X[idx_B])

  # Move the pairs to the device
  if device is not None:
    pairs = [t.to(device) for t in pairs]

  return pairs

def triplet_batch(batch_size, X, Y, encoded=False, size=None, train_embeddings=None, hardness=50, anchors=None, device=None):
  """
  Generates a batch of triplets (images or encoded samples) for training.

  Parameters
  ----------
  batch_size : int
      The number of triplets to generate in the batch.
  X : numpy.ndarray
      The input dataset containing images.
  Y : numpy.ndarray
      The class labels corresponding to each image in the dataset.
  encoded : bool, optional
      If True, the input `X` is already encoded and augmentation is skipped. Defaults to False.
  size : tuple, optional
      The desired image size to resize to. Only applicable if `X` contains image data. Defaults to None.
  train_embeddings : numpy.ndarray, optional
      The embeddings of the training data used for hard negative mining. Defaults to None.
  hardness : int, optional
      The percentile used to determine hard positive and negative examples. Defaults to 50.
  anchors : numpy.ndarray, optional
      The anchors to use for triplet generation. If None, random anchors are selected. Defaults to None.
  device : torch.device, optional
      The device to move the output tensors to (e.g., `torch.device('cuda')` for GPU). Defaults to None.

  Returns
  -------
  list of numpy.ndarray
      A list containing the generated triplets (anchor, positive, negative).
  """

  # Randomly select batch_size number of classes
  classes = np.random.choice(np.unique(Y), size=batch_size, replace=True)

  # Define the output shape based on the batch size, image size and encoding status
  if encoded:
    output_shape = (batch_size, X.shape[1])
  else:
    n_channels = 3 if len(X.shape) == 4 else 1
    if size is None:
      output_shape = (batch_size, n_channels, X.shape[1], X.shape[2])
    else:
      output_shape = (batch_size, n_channels, size[0], size[1])

  # Initialize arrays to store the pairs and their classes
  triplets = [torch.zeros(output_shape, dtype=torch.float32) for _ in range(3)]

  if not encoded:
    # Define the augmentation pipeline
    transform = A.Compose([
      A.RandomCrop(output_shape[2], output_shape[3]),
      A.VerticalFlip(),
      A.HorizontalFlip(),
      A.Rotate(),
      A.GaussianBlur(),
      A.RandomBrightnessContrast(),
      ToTensorV2()
    ])

  for i in range(batch_size):

    # Anchor mining
    # If anchors is provided, choose randomly from the anchors.
    # Otherwise, pick a random class and then a random sample from that class.
    if anchors is not None:
      anchor_index = np.random.choice(np.where(anchors)[0])
      anchor_class = Y[anchor_index]
      anchor_choices = np.where(Y == anchor_class)[0]

    else:
      anchor_class = classes[i]
      anchor_choices = np.where(Y == anchor_class)[0]
      anchor_index = np.random.choice(anchor_choices)

    # Offline triplet mining
    # When embeddings are available, use them to find hard positive and negative examples.
    # positive_dist measures the distance between the anchor and all positive examples.
    # Hard positives are those farther than a percentile defined by hardness.
    # Similarly, hard negatives are closer than the opposite percentile of hardness.
    if train_embeddings is not None:
      # Compute the distance of anchor_index to all positive examples
      positive_dist = np.linalg.norm(train_embeddings[anchor_index,:] - train_embeddings[anchor_choices,:], axis=1)
      # Get the percentile distance value, and only use the examples above it for creation of triplets
      valid_positive = positive_dist >= np.percentile(positive_dist, hardness)
      positive_index = np.random.choice(anchor_choices[valid_positive])

      # Compute the distance of anchor_index to all negative examples
      negative_choices = np.where(Y != anchor_class)[0]
      negative_dist = np.linalg.norm(train_embeddings[anchor_index,:] - train_embeddings[negative_choices,:], axis=1)
      valid_negative = negative_dist <= np.percentile(negative_dist, 100 - hardness)
      negative_index = np.random.choice(negative_choices[valid_negative])

    # Random triplets
    # When no embeddings are available, select random positive and negative examples.
    else:
      positive_index = np.random.choice(anchor_choices)

      negative_choices = np.where(Y != anchor_class)[0]
      negative_index = np.random.choice(negative_choices)

    if not encoded:
      anchor = transform(image=X[anchor_index])["image"]
      positive = transform(image=X[positive_index])["image"]
      negative = transform(image=X[negative_index])["image"]

      # Save the samples to the triplet list
      triplets[0][i] = anchor / 255.
      triplets[1][i] = positive / 255.
      triplets[2][i] = negative / 255.

    else:
      triplets[0][i] = torch.tensor(X[anchor_index])
      triplets[1][i] = torch.tensor(X[positive_index])
      triplets[2][i] = torch.tensor(X[negative_index])

  # Move the triplets to the device
  if device is not None:
    triplets = [t.to(device) for t in triplets]

  return triplets

class MulticlassDataset(torch.utils.data.Dataset):
  """
  Custom Dataset class for loading images and their corresponding labels.

  Parameters
  ----------
  X : numpy.ndarray
      The input dataset containing images.
  Y : numpy.ndarray
      The class labels corresponding to each image in the dataset.
  size : tuple, optional
      The desired image size to resize to. Defaults to None.
  encoded : bool, optional
      If True, the input `X` is already encoded and augmentation is skipped. Defaults to False.

  Methods
  -------
  __len__()
      Returns the number of images in the dataset.
  __getitem__(idx)
      Retrieves the image and its corresponding label at the specified index.
  """

  def __init__(self, X, Y, size=None, encoded=False):
    self.X = X
    self.Y = Y
    self.encoded = encoded

    # Define the output shape based on the batch size, image size and encoding status
    if encoded:
      output_shape = (X.shape[1])
    else:
      n_channels = 3 if len(X.shape) == 4 else 1
      if size is None:
        output_shape = (n_channels, X.shape[1], X.shape[2])
      else:
        output_shape = (n_channels, size[0], size[1])

    # Define augmentation pipeline
    self.transform = A.Compose([
      A.RandomCrop(output_shape[1], output_shape[2]),
      A.VerticalFlip(),
      A.HorizontalFlip(),
      A.Rotate(),
      A.GaussianBlur(),
      A.RandomBrightnessContrast(),
      ToTensorV2()
    ])

    self.test_transform = A.Compose([
      ToTensorV2()
    ])

  def __len__(self):
     return len(self.X)

  def __getitem__(self, idx):
    image = self.X[idx]
    label = self.Y[idx]

    if not self.encoded:
      image = self.transform(image = image)["image"]
    else:
      image = ToTensorV2()(image=image)["image"]

    image = image / 255.

    return image, label

#########################################################################################################
#     Some utility functions to convert images to PyTorch tensors and create a custom Dataset class     #
#########################################################################################################
def img_to_torch(batch, device=None):
  """
  Converts a batch of images to PyTorch tensors and optionally moves them to a specified device.

  Parameters
  ----------
  batch : list of numpy.ndarray
      A list of images where each image is a NumPy array.
  device : torch.device, optional
      The device to move the output tensors to (e.g., `torch.device('cuda')` for GPU). Defaults to None.

  Returns
  -------
  torch.Tensor
      A batch of images converted to a PyTorch tensor, optionally moved to the specified device.
  """
  
  # Define the transformations
  transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
  ])

  # Apply the transform to the batch of images
  batch = torch.stack([transform(im) for im in batch])

  if device is not None:
    # Move data to device
    batch = batch.to(device)

  return batch

class PatchData(torch.utils.data.Dataset):
  """
  A custom Dataset class for generating and retrieving image patches as PyTorch tensors.

  Parameters
  ----------
  data : numpy.ndarray
      The input dataset containing images from which patches will be generated.
  patch_size : tuple
      A tuple (height, width) indicating the size of the patches to be generated.
  device : torch.device, optional
      The device to move the output tensors to (e.g., `torch.device('cuda')` for GPU). Defaults to None.

  Methods
  -------
  __getitem__(index)
      Generates patches from the image at the specified index and converts them to PyTorch tensors.
  __len__()
      Returns the number of images in the dataset.
  """
  
  def __init__(self, data, patch_size, device=None):
    self.data = data
    self.device = device
    self.patch_size = patch_size
    self.size = self.data.shape[0]

  def __getitem__(self, index):
    patches = gen_patches(self.data[index], self.patch_size)
    return img_to_torch(patches, self.device)

  def __len__(self):
    return self.size
  

#########################################################################################################
#                                 Contrastive dissimilarity loss                                        #
#########################################################################################################
class DissimilarityNTXentLoss(torch.nn.Module):
  """
  Computes the Normalized Temperature-scaled Cross-Entropy (NT-Xent) loss for contrastive dissimilarity.

  Parameters
  ----------
  temperature : float, optional
      The temperature scaling factor for the softmax operation. Defaults to 0.5.

  Methods
  -------
  forward(diss, y)
      Computes the NT-Xent loss given the dissimilarity scores and labels.
  """
  def __init__(self, temperature=0.5):
    super(DissimilarityNTXentLoss, self).__init__()
    self.temperature = temperature

  def forward(self, diss, y):
    size = diss.shape[0]

    # Mask for positive samples
    y = torch.cat([y, y], dim=0)
    y1 = torch.tile(y, [size])
    y2 = torch.repeat_interleave(y, size, dim=0)
    pos_mask = torch.reshape(y1 == y2, (size, size))
    pos_mask.fill_diagonal_(False)

    # Mask for negative samples
    neg_mask = (~torch.eye(size, device=diss.device, dtype=bool)).float()

    # Compute nominator
    nominator = torch.sum(pos_mask * torch.exp(diss / self.temperature), dim=1)

    # Compute denominator
    denominator = torch.sum(neg_mask * torch.exp(diss / self.temperature), dim=1)

    # Compute loss
    loss_partial = -torch.log(nominator / denominator)
    loss = torch.mean(loss_partial)

    return loss
  

#########################################################################################################
#                                   Triplet dissimilarity loss                                          #
#########################################################################################################
class TripletDissimilarityLoss(torch.nn.Module):
  """
  Computes the triplet dissimilarity loss.

  Parameters
  ----------
  alpha : float
      The margin for the triplet loss.

  Methods
  -------
  forward(positive, negative)
      Computes the triplet dissimilarity loss given the positive and negative dissimilarity scores.
  """

  def __init__(self, alpha):
    super(TripletDissimilarityLoss, self).__init__()
    self.alpha = alpha

  def forward(self, positive, negative):
    loss_partial = torch.nn.functional.relu(positive - negative + self.alpha)
    loss = torch.mean(loss_partial)
    return loss


#########################################################################################################
#                                 Base network, projection head, and models                             #
#########################################################################################################
def _get_backbone(backbone, pretrained=True):
  """
  Get the backbone model and its feature size.

  Parameters
  ----------
  backbone : str
      The type of backbone model to use.
      As of now, tested with 'resnet50', 'resnet101', 'resnet152',
        'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l',
        'convnext_small', 'convnext_base', and 'convnext_large'.
  pretrained : bool or str, optional
      If False, no pre-trained weights are loaded.
      If True, uses default pre-trained weights.
      If a string, uses weights from the specified path.
      Defaults to True.

  Returns
  -------
  network : torch.nn.Module
      The backbone model.
  extra_layers : list
      A list of extra layers to be added to the model.
  feature_size : int
      The size of the feature vector output by the backbone model.
  """

  # Get the model function from torchvision
  model_fn = getattr(torchvision.models, backbone, None)

  # Load pretrained weights
  if isinstance(pretrained, str):
    raw_weights = torch.load(pretrained, weights_only=True)

    # Filter & remap keys
    weights = {}
    for k, v in raw_weights.items():
      # Drop classifier head weights entirely
      if k.startswith("network.classifier"):
        continue

      # Drop network prefix if present
      new_key = k
      if k.startswith("network."):
        new_key = k[len("network."):]

      weights[new_key] = v

    network = model_fn(weights=None)
    network.load_state_dict(weights, strict=False)
    print("Loaded pretrained weights from", pretrained)

  elif pretrained:
    network = model_fn(weights="DEFAULT")
    print("Loaded default pretrained weights from torchvision")
  else:
    network = model_fn(weights=None)

  extra_layers = []

  if hasattr(network, "fc"):
    feature_size = network.fc.in_features

  elif hasattr(network, "classifier"):
    cls = network.classifier
    if backbone.startswith("efficientnet"):
      feature_size = cls[1].in_features
    elif backbone.startswith("convnext"):
      # Keep LayerNorm2d + Flatten
      extra_layers.extend([cls[0], cls[1]])
      feature_size = cls[2].in_features
    else:
      raise RuntimeError(f"Unexpected classifier head for backbone {backbone!r}")
  else:
    raise RuntimeError(f"Could not find classifier head for backbone {backbone!r}")

  return network, extra_layers, feature_size


class Network(torch.nn.Module):
  """
  A base network for feature extraction.

  Parameters
  ----------
  embeddingsize : int
      The size of the output embedding vector.
  backbone : str
      Backbone model.
  pretrained : bool or str, optional
      If False, no pre-trained weights are loaded.
      If True, uses default pre-trained weights.
      If a string, uses weights from the specified path.
      Defaults to True.
  hidden_layers : list of int, optional
      A list of sizes for the hidden layers in the network. 
      If None, default hidden layers are used: [embeddingsize*4, embeddingsize*2].
      Defaults to None.
  num_classes : int, optional
      The number of classes for the classification head. Defaults to None.

  Methods
  -------
  forward(x)
      Passes the input through the shared network and normalizes the embedding vectors.
  """

  def __init__(self, embeddingsize, backbone, pretrained=True, hidden_layers=None, num_classes=None):
    super(Network, self).__init__()

    self.network, pre_head_layers, feature_size = _get_backbone(backbone, pretrained=pretrained)
    
    # Freeze the backbone network
    for param in self.network.parameters():
      param.requires_grad = False

    # Set the embedding head
    if hidden_layers is None:
      hidden_layers = [embeddingsize*4, embeddingsize*2]

    # Create the embedding head
    layers = []
    layers.extend(pre_head_layers)

    for hidden_size in hidden_layers:
      layers.append(torch.nn.Linear(feature_size, hidden_size))
      layers.append(torch.nn.ReLU())
      layers.append(torch.nn.Dropout(p=0.3))
      feature_size = hidden_size
    
    layers.append(torch.nn.Linear(feature_size, embeddingsize))

    # Replace the original classifier layers with the embedding head
    if hasattr(self.network, "fc"):
      self.network.fc = torch.nn.Sequential(*layers)
    else:
      self.network.classifier = torch.nn.Sequential(*layers)

    # If num_classes is provided, add a classifier head used for cross-entropy warmup
    if num_classes is not None:
      self.classifier = torch.nn.Sequential(
        torch.nn.Linear(embeddingsize, num_classes)
      )

  def forward(self, x, mode="embedding"):
    # Pass the input through the shared network
    x = self.network(x)

    # If the mode is set to "embedding", return the embedding vectors
    if mode == "embedding":
      # Normalize the embedding vectors
      return torch.nn.functional.normalize(x, p=2, dim=1)
    
    elif mode == "classifier":
      if self.classifier is None:
        raise RuntimeError("Classifier head not defined. Set num_classes to add a classifier head.")
      # Straight logits for cross-entropy
      return self.classifier(x)
    
    else:
      raise ValueError("Invalid mode. Choose either 'embedding' or 'classifier'.")


class ProjectionHead(torch.nn.Module):
  """
  Projection head for computing dissimilarity values.

  Parameters
  ----------
  embeddingsize : int
      The size of the output embedding vector.
  hidden_layers : list of int, optional
      Projection head hidden layers.
      If None, default hidden layers are used: [embeddingsize//2, embeddingsize//4].
      Defaults to None.
  output_size : int, optional
      The size of the output layer. Defaults to 1.

  Methods
  -------
  forward(x1, x2)
      Computes the dissimilarity between two input embeddings.
  """

  def __init__(self, embeddingsize, hidden_layers=None, output_size=1):
    super(ProjectionHead, self).__init__()

    # Set the hidden layers for the network
    if hidden_layers is None:
      hidden_layers = [embeddingsize//2, embeddingsize//4]

    # Define the projection head architecture dynamically
    layers = []
    input_size = embeddingsize

    for hidden_size in hidden_layers:
      layers.append(torch.nn.Linear(input_size, hidden_size))
      layers.append(torch.nn.ReLU())
      input_size = hidden_size
    
    layers.append(torch.nn.Linear(input_size, output_size))
    self.projection_head = torch.nn.Sequential(*layers)

  def forward(self, x1, x2):
    return self.projection_head(torch.abs(x1 - x2))


class ContrastiveModel(torch.nn.Module):
  """
  Contrastive dissimilarity model.

  Parameters
  ----------
  embeddingsize : int
      The size of the output embedding vector.
  backbone : str
      Backbone model.
  projection_head : list of int, optional
      Projection head hidden layers.
      If None, default hidden layers are used: [embeddingsize//2, embeddingsize//4].
      Defaults to None.
  top_layers : list of int, optional
      Base network top layers.
      If None, default hidden layers are used: [embeddingsize*4, embeddingsize*2].
      Defaults to None.
  encoded : bool, optional
      If True, the input is already encoded, and the model does not need to estimate embeddings.
      Defaults to False.
  pretrained : bool or str, optional
      If False, no pre-trained weights are loaded.
      If True, uses default pre-trained weights.
      If a string, uses weights from the specified path.
      Defaults to True.
  num_classes : int, optional
      The number of classes for the classification head. Defaults to None.

  Methods
  -------
  forward(x1, x2)
      Computes the dissimilarity between pairs of input images.
  freeze_network()
      Freezes the parameters of the base network to prevent them from being updated during training.
  unfreeze_network()
      Unfreezes the parameters of the base network to allow them to be updated during training.
  """

  def __init__(self, embeddingsize, backbone, projection_head=None, top_layers=None, encoded=False, pretrained=True, num_classes=None):
    super(ContrastiveModel, self).__init__()

    # Check if encoded is set to True
    # If so, the model does not need to estimate embeddings
    self.network = None
    if not encoded:
      self.network = Network(embeddingsize, backbone=backbone, hidden_layers=top_layers, pretrained=pretrained, num_classes=num_classes)

    self.projection_head = ProjectionHead(embeddingsize, hidden_layers=projection_head)

  def forward(self, x1, x2):

    # Encode the inputs
    if self.network is not None:
      x1 = self.network(x1)
      x2 = self.network(x2)

    if self.training:
      batch_size = x1.shape[0]

      # Repeat the elements to match the input expected by the network
      x = torch.cat([x1, x2])
      x1 = torch.tile(x, [batch_size * 2, 1])
      x2 = torch.repeat_interleave(x, batch_size * 2, dim=0)

      dissimilarity = self.projection_head(x1, x2)
      dissimilarity = torch.reshape(dissimilarity, (batch_size * 2, -1))
    else:
      dissimilarity = self.projection_head(x1, x2)

    return dissimilarity

  def freeze_network(self):
    if self.network is not None:
      for param in self.network.parameters():
        param.requires_grad = False

  def unfreeze_network(self):
    if self.network is not None:
      for param in self.network.parameters():
        param.requires_grad = True


class TripletModel(torch.nn.Module):
  """
  Triplet dissimilarity model.

  Parameters
  ----------
  embeddingsize : int
      The size of the output embedding vector.
  backbone : str
      Backbone model.
  projection_head : list of int, optional
      Projection head hidden layers.
      If None, default hidden layers are used: [embeddingsize//2, embeddingsize//4].
      Defaults to None.
  top_layers : list of int, optional
      Base network top layers.
      If None, default hidden layers are used: [embeddingsize*4, embeddingsize*2].
      Defaults to None.
  encoded : bool, optional
      If True, the input is already encoded, and the model does not need to estimate embeddings.
      Defaults to False.
  pretrained : bool or str, optional
      If False, no pre-trained weights are loaded.
      If True, uses default pre-trained weights.
      If a string, uses weights from the specified path.
      Defaults to True.
  num_classes : int, optional
      The number of classes for the classification head. Defaults to None.

  Methods
  -------
  forward(anchor, positive, negative)
      Computes the dissimilarity between triplet inputs.
  freeze_network()
      Freezes the parameters of the base network to prevent them from being updated during training.
  unfreeze_network()
      Unfreezes the parameters of the base network to allow them to be updated during training.
  """

  def __init__(self, embeddingsize, backbone, projection_head=None, top_layers=None, encoded=False, pretrained=True, num_classes=None):
    super(TripletModel, self).__init__()

    # Check if encoded is set to True
    # If so, the model does not need to estimate embeddings
    self.network = None
    if not encoded:
      self.network = Network(embeddingsize, backbone=backbone, hidden_layers=top_layers, pretrained=pretrained, num_classes=num_classes)

    self.projection_head = ProjectionHead(embeddingsize, hidden_layers=projection_head)

  def forward(self, anchor, positive, negative):

    # Encode the inputs
    if self.network is not None:
      anchor = self.network(anchor)
      positive = self.network(positive)
      negative = self.network(negative)

    # Compute dissimilarity scores
    pos_dissimilarity = self.projection_head(anchor, positive)
    neg_dissimilarity = self.projection_head(anchor, negative)

    return pos_dissimilarity, neg_dissimilarity

  def freeze_network(self):
    if self.network is not None:
      for param in self.network.parameters():
        param.requires_grad = False

  def unfreeze_network(self):
    if self.network is not None:
      for param in self.network.parameters():
        param.requires_grad = True


def train(X, Y, model_type, model_file, backbone,
          
          # Common model parameters
          embeddingsize=128, patch_size=None, projection_head=None, top_layers=None, pretrained=True,
          batch=32, iterations=10000, lr=0.001,

          # Common warmup parameters
          batch_warmup=64,

          # Cross-entropy warmup parameters
          clf_warmup=False, clf_warmup_epochs=20, clf_epochs=50, clf_warmup_lr=0.01, clf_lr=0.001,

          # Projection head warmup parameters
          warmup_iterations=1000, lr_warmup=0.01,

          # Contrastive model parameters
          temperature_warmup=0.5, temperature=0.5, 

          # Triplet model parameters
          alpha_warmup=1.0, alpha=1.0, 
          triplet_mining=False, mining_iterations=10000, mining_hardness=50, mining_lr=0.001):
  """
  Train a contrastive or triplet model for image classification.

  Parameters
  ----------
  X : numpy.ndarray
      The input dataset containing images.
  Y : numpy.ndarray
      The class labels corresponding to each image in the dataset.
  model_type : str
      The type of model to train. Either 'contrastive' or 'triplet'.
  model_file : str
      The file path to save the trained model.
  backbone : str
      The backbone model to use for feature extraction.
  embeddingsize : int, optional
      The size of the output embedding vector. Defaults to 128.
  patch_size : tuple, optional
      The size of the patches to be generated from the images. Defaults to None.
  projection_head : list of int, optional
      The sizes of the hidden layers in the projection head. Defaults to None.
  top_layers : list of int, optional
      The sizes of the hidden layers in the base network. Defaults to None.
  pretrained : bool or str, optional
      If False, no pre-trained weights are loaded.
      If True, uses default pre-trained weights.
      If a string, uses weights from the specified path.
      Defaults to True.
  batch : int, optional
      The batch size for training. Defaults to 32.
  iterations : int, optional
      The number of training iterations. Defaults to 10000.
  lr : float, optional
      The learning rate for the optimizer. Defaults to 0.001.
  batch_warmup : int, optional
      The batch size for the warmup phase. Defaults to 64.
  clf_warmup : bool, optional
      If True, performs a cross-entropy warmup phase. Defaults to False.
  clf_warmup_epochs : int, optional
      The number of epochs for the cross-entropy warmup phase. Defaults to 50.
  clf_epochs : int, optional
      The number of epochs for the main cross-entropy training phase. Defaults to 200.
  clf_warmup_lr : float, optional
      The learning rate for the cross-entropy warmup phase. Defaults to 0.01.
  clf_lr : float, optional
      The learning rate for the main cross-entropy training phase. Defaults to 0.001.
  warmup_iterations : int, optional
      The number of iterations for the projection head warmup phase. Defaults to 1000.
  lr_warmup : float, optional
      The learning rate for the projection head warmup phase. Defaults to 0.01.
  temperature_warmup : float, optional
      The temperature for the warmup phase in the contrastive model. Defaults to 0.5.
  temperature : float, optional 
      The temperature for the main training phase in the contrastive model. Defaults to 0.5.
  alpha_warmup : float, optional
      The margin for the warmup phase in the triplet model. Defaults to 1.0.
  alpha : float, optional
      The margin for the main training phase in the triplet model. Defaults to 1.0.
  triplet_mining : bool, optional
      If True, performs triplet mining during training. Defaults to False.
  mining_iterations : int, optional
      The number of iterations for triplet mining. Defaults to 10000.
  mining_hardness : int, optional
      The hardness level for triplet mining. Defaults to 50.
  mining_lr : float, optional
      The learning rate for the triplet mining phase. Defaults to 0.001.

  Returns
  -------
  torch.nn.Module
      The trained model.
  """
  
  print(f"Model file: {model_file}")

  # Define computation device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = None

  # Check if embeddingsize is set to None
  # If so, the model does not need to estimate embeddings, the input is already encoded
  # The embeddingsize is set to the number of features in the input data
  encoded = False
  if embeddingsize is None:
    encoded = True
    embeddingsize = X.shape[1]

  # Apply a label encoder
  # This is needed for the cross-entropy warmup
  # It makes no difference for the dissimilarity model
  Y = LabelEncoder().fit_transform(Y)

  # Get the number of classes
  num_classes = len(np.unique(Y))

  if model_type == "contrastive":
    model_fn = ContrastiveModel
    loss_warmup_fn = DissimilarityNTXentLoss(temperature_warmup)
    loss_fn = DissimilarityNTXentLoss(temperature)
  elif model_type == "triplet":
    model_fn = TripletModel
    loss_warmup_fn = TripletDissimilarityLoss(alpha_warmup)
    loss_fn = TripletDissimilarityLoss(alpha)
  else:
    raise ValueError("Invalid model type. Choose either 'contrastive' or 'triplet'.")

  # Load pre-trained model
  if os.path.isfile(model_file):
    print("Loading pre-trained model...")
    model = model_fn(embeddingsize, backbone,
                     projection_head=projection_head, top_layers=top_layers, encoded=encoded, 
                     pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_file, weights_only=True))
    model.to(device)

  # Train a new model if not loaded
  if model is None:
    print("Training a new model...")

    # Create the model
    model = model_fn(embeddingsize, backbone, 
                     projection_head=projection_head, top_layers=top_layers, encoded=encoded, 
                     pretrained=pretrained, num_classes=num_classes)

    model.to(device)
    model.train()

    #######################################################
    #                 Cross-entropy warmup                #
    #######################################################
    if clf_warmup:
      print("Cross-entropy Warmup Phase")

      # Get the backbone model
      backbone = model.network

      # Set the data
      dataset = MulticlassDataset(X, Y, size=patch_size, encoded=encoded)
      train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_warmup, shuffle=True)

      # Initialize loss, optimizer, and training mode
      criterion = torch.nn.CrossEntropyLoss()
      optimizer = torch.optim.SGD(backbone.parameters(), lr=clf_warmup_lr, momentum=0.9)

      print("Warmup top-layers")
      epoch_loss = 0
      for epoch in range(clf_warmup_epochs):
        for images, labels in train_loader:
          images = images.to(device)
          labels = labels.to(device)

          # Zero the gradients
          optimizer.zero_grad()

          # Forward pass
          outputs = backbone(images, mode="classifier")

          # Compute loss
          loss = criterion(outputs, labels)
          epoch_loss += loss.item()

          # Backward pass
          loss.backward()
          optimizer.step()

        # Compute average epoch loss
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.4f}")
        epoch_loss = 0

      print("Warmup backbone")

      # Unfreeze the whole model, including the backbone
      model.unfreeze_network()

      # Reinitialize optimizer for the training phase with a lower learning rate
      optimizer = torch.optim.SGD(model.parameters(), lr=clf_lr, momentum=0.9)

      # Training phase
      epoch_loss = 0
      for epoch in range(clf_epochs):
        for images, labels in train_loader:
          images = images.to(device)
          labels = labels.to(device)

          # Zero the gradients
          optimizer.zero_grad()

          # Forward pass
          outputs = backbone(images, mode="classifier")

          # Compute loss
          loss = criterion(outputs, labels)
          epoch_loss += loss.item()

          # Backward pass
          loss.backward()
          optimizer.step()

        # Compute average epoch loss
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.4f}")
        epoch_loss = 0

      # Freeze the backbone network again
      model.freeze_network()

    #######################################################
    #         Projection head/top-layers warmup           #
    #######################################################
    if pretrained:
      print("Projection Head Warmup Phase")

      optimizer = torch.optim.SGD(model.parameters(), lr=lr_warmup, momentum=0.9)

      train_loss = 0
      for epoch in range(warmup_iterations // 100):
        for _ in range(100):

          optimizer.zero_grad()

          if model_type == "contrastive":
            x1, x2, y = pair_batch(batch_warmup, X, Y, encoded=encoded, size=patch_size, device=device)
            outputs = model(x1, x2)
            loss = loss_warmup_fn(outputs, y)
          elif model_type == "triplet":
            anc, pos, neg = triplet_batch(batch_warmup, X, Y, encoded=encoded, size=patch_size, device=device)
            pos_score, neg_score = model(anc, pos, neg)
            loss = loss_warmup_fn(pos_score, neg_score)
          else:
            raise ValueError("Invalid model type. Choose either 'contrastive' or 'triplet'.")

          train_loss += loss.item()

          # Backward pass and optimization
          loss.backward()
          optimizer.step()

        # Compute average loss for the epoch
        print(f"Epoch {epoch + 1}, Warmup Loss: {train_loss / 100:.4f}")
        train_loss = 0      

    #######################################################
    #                Full training phase                  #
    #######################################################
    print("Training Phase")

    # Unfreeze the network for full training
    model.unfreeze_network()

    # Reinitialize optimizer for the training phase with a lower learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Training phase
    train_loss = 0
    for epoch in range(iterations // 100):
      for _ in range(100):
        
        optimizer.zero_grad()

        if model_type == "contrastive":
          x1, x2, y = pair_batch(batch, X, Y, encoded=encoded, size=patch_size, device=device)
          outputs = model(x1, x2)
          loss = loss_fn(outputs, y)
        elif model_type == "triplet":
          anc, pos, neg = triplet_batch(batch, X, Y, encoded=encoded, size=patch_size, device=device)
          pos_score, neg_score = model(anc, pos, neg)
          loss = loss_fn(pos_score, neg_score)
        else:
          raise ValueError("Invalid model type. Choose either 'contrastive' or 'triplet'.")
          
        train_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

      # Compute average loss for the epoch
      print(f"Epoch {epoch + 1}, Training Loss: {train_loss / 100:.4f}")
      train_loss = 0

    # Save the trained model
    print("Saving the trained model...")
    # Extract the directory from the save location
    dir = os.path.dirname(model_file)
    if dir:
      os.makedirs(dir, exist_ok=True)
    torch.save(model.state_dict(), model_file)


  #########################################################
  #                Triplet mining phase                   #
  #########################################################
  if model_type == "triplet" and triplet_mining:
    print("Triplet Mining Phase")

    # Extract patches from the training data
    # Use the same patch size as in the training phase
    print("Extracting patches from training data...")
    model.eval()
    with torch.no_grad():
      patch_data = PatchData(X, patch_size=patch_size, device=device)
      patch_dataloader = torch.utils.data.DataLoader(dataset=patch_data, batch_size=None, shuffle=False)
      train_embeddings = torch.stack([model.network(data) for _, data in enumerate(patch_dataloader)])
    
    # Convert to numpy and compute the mean for each sample
    train_embeddings = np.array(train_embeddings.cpu(), dtype=np.float32)
    train_embeddings = np.mean(train_embeddings, axis=1)

    # Find useful anchors for triplet mining
    # Useful anchors are hard observations, i.e., they are close to negative examples
    print("Finding useful anchors for triplet mining...")
    n_obs = train_embeddings.shape[0]
    anchors = np.zeros(n_obs, dtype=bool)

    for anchor_index in range(n_obs):
      anchor_class = Y[anchor_index]
      negative_choices = np.where(Y != anchor_class)[0]
      negative_dist = np.linalg.norm(train_embeddings[anchor_index,:] - train_embeddings[negative_choices,:], axis=1)
      valid_negative = negative_dist <= alpha
      anchors[anchor_index] = np.any(valid_negative)

    if not np.any(anchors):
      print("No anchors found, try increasing alpha")
      return None, None
    
    # Instantiate the triplet loss
    loss_fn = TripletDissimilarityLoss(alpha)

    # Reinitialize optimizer for the triplet mining phase with a lower learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=mining_lr, momentum=0.9)

    # Triplet mining phase
    model.train()
    train_loss = 0
    for epoch in range(mining_iterations // 100):
      for _ in range(100):

        optimizer.zero_grad()

        anc, pos, neg = triplet_batch(batch, X, Y, encoded=encoded, size=patch_size, 
                                      train_embeddings=train_embeddings, hardness=mining_hardness, anchors=anchors, device=device)
        pos_score, neg_score = model(anc, pos, neg)
        loss = loss_fn(pos_score, neg_score)

        train_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

      # Compute average loss for the epoch
      print(f"Epoch {epoch + 1}, Training Loss: {train_loss / 100:.4f}")
      train_loss = 0

    # Save the trained model
    print("Saving the trained model...")
    # Extract the directory from the save location
    dir = os.path.dirname(model_file)
    if dir:
      os.makedirs(dir, exist_ok=True)
    torch.save(model.state_dict(), model_file)

  # Freeze the model parameters and set to evaluation mode
  model.freeze_network()
  model.eval()

  print("Model is ready for evaluation.")

  return model


#########################################################################################################
#                                       Embedding generation                                            #
#########################################################################################################
def generate_embedding(model, data, patch_size, cache="embedding.pkl"):
  """
  Generate embeddings for a given dataset and caches them.

  Parameters
  ----------
  model : torch.nn.Module
      The trained model to be used for generating embeddings.
  data : numpy.ndarray
      The data as a NumPy array.
  patch_size : tuple of int
      The size of the patches to be generated from the images.
  cache : str, optional
      The file path for caching and loading precomputed embeddings. Defaults to 'embedding.pkl'.
      Also accepts False to disable caching.

  Returns
  -------
  np.ndarray
      The computed embeddings for the specified dataset.
  """

  # Check if the embeddings file exists
  if cache is not False and os.path.isfile(cache):
    with open(cache, "rb") as f:
      embedding = pickle.load(f)
    return embedding

  # Define computation device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Set up the patch loader
  patch_data = PatchData(data, patch_size=patch_size, device=device)
  patch_dataloader = torch.utils.data.DataLoader(dataset=patch_data, batch_size=None, shuffle=False)

  # Extract patches and generate embeddings
  embedding = torch.stack([
    model.network(batch_data) for _, batch_data in enumerate(patch_dataloader)
  ])

  # Convert to numpy and compute the mean for each sample
  embedding = np.array(embedding.cpu(), dtype=np.float32)
  embedding = np.mean(embedding, axis=1)

  # Store the embeddings and save them to the cache file
  if cache is not False:
    with open(cache, "wb") as f:
      pickle.dump(embedding, f, protocol=pickle.HIGHEST_PROTOCOL)

  return embedding


#########################################################################################################
#                                            UMAP                                                       #
#########################################################################################################
def umap_projection(encoded_X, Y, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
  """
  Visualizes high-dimensional encodings using UMAP for dimensionality reduction.

  Parameters
  ----------
  encoded_X : np.ndarray
      The high-dimensional encodings to be visualized, of shape (num_samples, num_features).
  Y : np.ndarray or list
      The labels or classes corresponding to each sample in encoded_X.
  n_neighbors : int, optional
      The number of neighbors to consider for UMAP. Controls local versus global structure. Defaults to 15.
  min_dist : float, optional
      The minimum distance between points in the low-dimensional UMAP representation. Defaults to 0.1.
  n_components : int, optional
      The number of dimensions for the UMAP output. Typically 2 for 2D visualization. Defaults to 2
  random_state : int, optional
      The random seed for reproducibility. Defaults to 42.

  Returns
  -------
  None
  """

  # Check input dimensions
  assert len(encoded_X) == len(Y), "The number of samples in encoded_X and Y must be the same."

  # Initialize UMAP reducer
  reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state, n_jobs=1)
  
  # Fit and transform the data
  trn = reducer.fit_transform(encoded_X)

  # Plotting
  plt.figure(figsize=(10, 6))
  sns.scatterplot(
    x=trn[:, 0], 
    y=trn[:, 1], 
    hue=Y, 
    palette=sns.color_palette("bright", len(np.unique(Y))), 
    legend=False
  )
  plt.title("UMAP Projection")
  plt.xlabel("UMAP-1")
  plt.ylabel("UMAP-2")
  plt.show()


#########################################################################################################
#                                             Prototype selection                                       #
#########################################################################################################
def compute_centroids(X, Y, K):
  """
  Computes centroids for each cluster.

  Parameters
  ----------
  X : np.ndarray
      The input data as a NumPy array of shape (num_samples, num_features).
  Y : np.ndarray
      The cluster labels corresponding to each sample in X.
  K : int
      The number of clusters.

  Returns
  -------
  np.ndarray
      A NumPy array of shape (K, num_features) containing the centroids of each cluster.
  """

  m, n = X.shape
  centroids = np.zeros((K, n))
  for k in range(K):
    x = X[Y == k]
    centroids[k, :] = np.mean(x, axis=0)
  return centroids

def compute_prototypes(embeddings, Y, n_prototypes=5, method="kmeans", cache="prototypes.pkl"):
  """
  Computes prototypes for each class using specified clustering methods.

  Parameters
  ----------
  embeddings : np.ndarray
      The embeddings data as a NumPy array of shape (num_samples, num_features).
  Y : np.ndarray
      The class labels corresponding to each sample in the embeddings array.
  n_prototypes : int, optional
      The number of prototypes to compute for each class. Defaults to 5.
  method : str, optional
      The clustering method to use for computing prototypes. Options are 'kmeans', 'spectral', and 'hierarchical'. Defaults to 'kmeans'.
  cache : str, optional
      The file path for caching precomputed prototypes. Defaults to 'prototypes.pkl'.
      Also accepts False to disable caching.

  Returns
  -------
  np.ndarray
      A NumPy array of shape (total_prototypes, num_features) containing the computed prototypes for each class.
  np.ndarray
      A NumPy array of shape (total_prototypes,) containing the class label for each prototype.
  """

  # Check if the prototypes file exists
  if os.path.isfile(cache):
    with open(cache, "rb") as f:
      prototypes, classes = pickle.load(f)
    return prototypes, classes

  # Count the number of unique classes in the data
  uniq_classes = np.unique(Y)
  n_classes = len(uniq_classes)

  # Compute the total number of prototypes
  total_prototypes = n_classes * n_prototypes
  prototypes = np.zeros((total_prototypes, embeddings.shape[1]), dtype=np.float32)
  classes = np.zeros(total_prototypes, dtype=np.int32)

  # Find prototypes in each class
  for idx, cls in enumerate(uniq_classes):
    X_embedding = embeddings[np.where(Y == cls)]
    start, end = n_prototypes * (idx + 1) - n_prototypes, n_prototypes * (idx + 1)

    # Clustering based on the chosen method
    if method == "kmeans":
      # K-means clustering to find prototypes
      clustering = KMeans(n_clusters=n_prototypes, init="k-means++", n_init="auto", random_state=1234).fit(X_embedding)
      centroids = clustering.cluster_centers_
    elif method == "spectral":
      # Spectral clustering
      clustering = SpectralClustering(n_clusters=n_prototypes, affinity="nearest_neighbors", random_state=1234).fit(X_embedding)
      labels = clustering.labels_
      centroids = compute_centroids(X_embedding, labels, n_prototypes)
    elif method == "hierarchical":
      # Hierarchical clustering
      clustering = AgglomerativeClustering(n_clusters=n_prototypes).fit(X_embedding)
      labels = clustering.labels_
      centroids = compute_centroids(X_embedding, labels, n_prototypes)
    else:
      raise ValueError("Unsupported clustering method. Choose from 'kmeans', 'spectral', or 'hierarchical'.")

    # Store the computed centroids and assign class labels to each prototype
    prototypes[start:end, :] = centroids
    classes[start:end] = cls  # Assign the class label to each prototype

  # Save the computed prototypes and their classes to a file
  if cache is not False:
    with open(cache, "wb") as f:
      pickle.dump((prototypes, classes), f, protocol=pickle.HIGHEST_PROTOCOL)

  return prototypes, classes


#########################################################################################################
#                              Metric Dissimilarity representation                                      #
#########################################################################################################
def space_representation(model, encoded, X_prot, cache="space.pkl"):
  """
  Computes the metric dissimilarity space for a given dataset.

  Parameters
  ----------
  model : torch.nn.Module
      The trained model with a projection head for computing dissimilarity.
  encoded : np.ndarray
      A NumPy array containing the encoded data for the specific dataset to compute dissimilarity for.
  X_prot : np.ndarray
      A NumPy array of shape (num_prototypes, num_features) containing the prototypes.
  cache : str, optional
      The file path for caching precomputed dissimilarity space. Defaults to 'space.pkl'.
      Also accepts False to disable caching.

  Returns
  -------
  np.ndarray
      A NumPy array containing the dissimilarity space representation.
  """

  # Check if the space file exists
  if cache is not False and os.path.isfile(cache):
    with open(cache, "rb") as f:
      space = pickle.load(f)
    return space

  # Convert prototypes to PyTorch tensor and move to GPU
  X_prot = torch.from_numpy(X_prot).to("cuda")
  n_prot = X_prot.shape[0]

  # Initialize the space list for the given set
  space = []

  # Loop through each data point in the dataset
  for idx in range(encoded.shape[0]):
    local_x = encoded[[idx], :].astype(np.float32)
    local_x = np.repeat(local_x, n_prot, axis=0)  # Repeat to match the number of prototypes
    local_x = torch.from_numpy(local_x).to("cuda")
    
    # Compute dissimilarity using the model's projection head
    diss = model.projection_head(local_x, X_prot).squeeze().cpu().detach().numpy()
    space.append(diss)

  # Convert the list to a NumPy array for consistency
  space = np.array(space)

  # Save the computed dissimilarity space to a file
  if cache is not False:
    with open(cache, "wb") as f:
      pickle.dump(space, f, protocol=pickle.HIGHEST_PROTOCOL)

  return space


def vector_representation(model, X, Y, X_prot, Y_prot, patch_size=None, variations=20, cache="vector.pkl"):
  """
  Computes the metric dissimilarity vector for given datasets using a projection head model.

  Parameters
  ----------
  model : torch.nn.Module
      The trained model with a projection head for computing dissimilarity.
  X : np.ndarray
      The input data as a NumPy array.
  Y : np.ndarray
      The class labels for the input data.
  X_prot : np.ndarray
      The prototypes data as a NumPy array.
  Y_prot : np.ndarray
      The class labels for the prototype data.
  patch_size : tuple, optional
      The size of the patches to be generated from the images. Defaults to None.
  variations : int, optional
      The minimum number of variations to generate per input. Defaults to 20.
  cache : str, optional
      The file path for caching precomputed dissimilarity vector. Defaults to 'vector.pkl'.
      Also accepts False to disable caching.

  Returns
  -------
  dict
      A dictionary containing the dissimilarity vector representation and corresponding labels for input data.
  """

  # Check if the vector file exists
  if cache is not False and os.path.isfile(cache):
    with open(cache, "rb") as f:
      X_vector, Y_vector = pickle.load(f)
    return X_vector, Y_vector

  # Generate label pairs for training data
  Y_vector = np.equal.outer(Y, Y_prot).ravel()

  X_vector = []

  # If the network was not trained, the input was already encoded
  # Thus we are not dealing with images
  # Enable training mode to activate dropout
  # This way the same input generates slightly different outputs that we treat as "augmentations"
  if model.network is None:
    model.train()

  # Loop through each data point in the dataset
  for idx in range(X.shape[0]):

    # Prepare patches and prototypes for the projection head
    if model.network is None:
      number_prototypes = X_prot.shape[0]
      
      local_encodings = torch.from_numpy(np.float32(X[idx])).to("cuda")
      local_encodings = torch.tile(local_encodings, [number_prototypes * variations, 1])

      local_prototypes = np.repeat(X_prot, variations, axis=0)
      local_prototypes = torch.from_numpy(local_prototypes).to("cuda")
    
    else:
      local_patches = img_to_torch(gen_patches(X[idx], patch_size, min_patches=variations * 5, regular=False), device="cuda")
      patch_encodings = model.network(local_patches)

      # Get the mean of a set of patches to create more stable encodings
      patch_encodings = torch.mean(torch.stack(patch_encodings.split(5)), dim=1, dtype=torch.float32)

      number_patches = patch_encodings.shape[0]
      number_prototypes = X_prot.shape[0]

      local_encodings = torch.tile(patch_encodings, [number_prototypes, 1])
      local_prototypes = np.repeat(X_prot, number_patches, axis=0)
      local_prototypes = torch.from_numpy(local_prototypes).to("cuda")

    diss_vec = model.projection_head(local_encodings, local_prototypes)
    diss_vec = np.array(np.split(diss_vec.cpu().detach().numpy(), number_prototypes))
    X_vector.append(diss_vec)

  # Reshape to match the labels
  X_vector = np.reshape(X_vector, (len(Y) * number_prototypes, -1))

  # Save the computed dissimilarity vectors to a file
  if cache is not False:
    with open(cache, "wb") as f:
      pickle.dump((X_vector, Y_vector), f, protocol=pickle.HIGHEST_PROTOCOL)

  return X_vector, Y_vector


def vector_to_class(X_proba, Y, Y_prot):
  """
  Transforms metric dissimilarity vector representation back into multiclass classification.

  Parameters
  ----------
  X_proba : np.ndarray
      The predicted probabilities for the test data as an array of shape (n_samples * n_prototypes, 2).
  Y : np.ndarray
      The true class labels for the test data.
  Y_prot : np.ndarray
      The prototype labels used to determine the number of prototypes per class.

  Returns
  -------
  np.ndarray
      The predicted class labels for the test data.
  """
  
  # Reshape to match the number of test samples
  X_proba = np.reshape(X_proba[:, 1], (Y.shape[0], -1)) # (n_samples, n_prototypes)

  # Find the number of prototypes per class, a single int value
  prot_per_class = np.bincount(Y_prot).max()

  # Average the prediction probabilities across all prototypes
  X_proba = np.reshape(X_proba, (Y.shape[0], -1, prot_per_class)) # (n_samples, n_prototypes, n_prototypes_per_class)
  X_proba = np.max(X_proba, axis=-1) # (n_samples, n_prototypes)

  # Get the class with the highest probability for each test sample
  X_pred = np.argmax(X_proba, axis=1) # (n_samples,)

  return X_pred


#########################################################################################################
#                             Traditional dissimilarity representation                                  #
#########################################################################################################
def cosine_distance(x, y):
  """
  Computes the cosine distance between two sets of vectors.

  Parameters
  ----------
  x : np.ndarray
      The first set of vectors.
  y : np.ndarray
      The second set of vectors.

  Returns
  -------
  np.ndarray
      The cosine distance between the two sets of vectors.
  """

  # Normalize the vectors
  norm_x = x / np.linalg.norm(x, axis=1, keepdims=True)
  norm_y = y / np.linalg.norm(y, axis=1, keepdims=True)
  
  # Compute the cosine distance
  return np.matmul(norm_x, norm_y.T)

def tradt_space_representation(encoded, X_prot, distance="euclidean", cache="tradt-space.pkl"):
  """
  Computes the traditional dissimilarity space.

  Parameters
  ----------
  encoded : np.ndarray
      A NumPy array containing the encoded data for the specific dataset to compute dissimilarity for.
  X_prot : np.ndarray
      A NumPy array of shape (num_prototypes, num_features) containing the prototypes.
  distance : str, optional
      The distance metric to use for dissimilarity computation. Can be either "euclidean" or "cosine". Defaults to "euclidean".
  cache : str, optional
      The file path for caching precomputed dissimilarity space. Defaults to 'tradt-space.pkl'.
      Also accepts False to disable caching.

  Returns
  -------
  np.ndarray
      A NumPy array containing the dissimilarity space representation.
  """

  # Check if the traditional dissimilarity space file exists
  if cache is not False and os.path.isfile(cache):
    with open(cache, "rb") as f:
      space = pickle.load(f)
    return space

  if distance == "euclidean":
    # Create a new axis to facilitate broadcasting
    encoded = encoded[:,np.newaxis,:] # (n,1,d)

    # Compute the Euclidean distance
    space = np.linalg.norm(encoded - X_prot, axis=2) # (n, m)

  elif distance == "cosine":
    # Compute the cosine distance
    space = cosine_distance(encoded, X_prot) # (n, m)
    
  else:
    raise ValueError("Unsupported distance metric. Choose either 'euclidean' or 'cosine'.")
  
  # Save the computed traditional dissimilarity space to a file
  if cache is not False:
    with open(cache, "wb") as f:
      pickle.dump(space, f, protocol=pickle.HIGHEST_PROTOCOL)

  return space


def tradt_vector_representation(encoded, Y, X_prot, Y_prot, cache="tradt-vector.pkl"):
  """
  Computes the traditional dissimilarity vector.

  Parameters
  ----------
  encoded : np.ndarray
      A NumPy array containing the encoded data for the specific dataset to compute dissimilarity for.
  Y : np.ndarray
      The class labels for the input data.
  X_prot : np.ndarray
      The prototypes data as a NumPy array.
  Y_prot : np.ndarray
      The class labels for the prototype data.
  cache : str, optional
      The file path for caching precomputed dissimilarity vector. Defaults to 'tradt-vector.pkl'.
      Also accepts False to disable caching.

  Returns
  -------
  np.ndarray
      A NumPy array containing the dissimilarity vector representation.
  np.ndarray
      A NumPy array containing the corresponding labels for the dissimilarity vector.
  """

  # Check if the traditional dissimilarity vector file exists
  if cache is not False and os.path.isfile(cache):
    with open(cache, "rb") as f:
      X_vector, Y_vector = pickle.load(f)
    return X_vector, Y_vector

  # Same-label indicator for every pair
  Y_vector = np.equal.outer(Y, Y_prot).ravel()

  # Get the embedding size
  embedding = encoded.shape[1]

  # Create a new axis to facilitate broadcasting
  encoded = encoded[:,np.newaxis,:] # (n,1,d)
  X_prot = X_prot[np.newaxis, :, :] # (1,m,d)

  X_vector = np.abs(encoded - X_prot).reshape(-1, embedding) # (n*m, d)

  # Save the computed traditional dissimilarity vector to a file
  if cache is not False:
    with open(cache, "wb") as f:
      pickle.dump((X_vector, Y_vector), f, protocol=pickle.HIGHEST_PROTOCOL)

  return X_vector, Y_vector