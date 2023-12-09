
# Custom Dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_file, labels_file, transform=None):
        self.data = np.load(data_file)
        self.labels = np.load(labels_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[:, idx].reshape(300, 300, 3)  # Reshape the individual image to its original shape
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)


        return image, label


# Model architecture
import torch
import torch.nn as nn
import torch.nn.functional as F


# Filter/kernel dimensions of 3 x 3.
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# Filter/kernel dimensions of 1 x 1.
# 1 x 1 convolutions act as "bottleneck" layers, which reduce the number of
# parameters to a reasonable size such that computational costs are
# decreased.
# Number of parameters in a given layer are derived only from the dimensions
# of the filter, not from the dimensions of the input image.
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# According to the paper, each residual block should consist of two
# 3 x 3 convolutions with batch normalization and skip connections.
# These residual blocks are fed into the ResNet class.
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# Main class that takes in a given residual block, the corresponding
# number of layers and the number of classes.
# Residual blocks are called here, as ResNet consists of a combination
# of convolutional, residual and bottleneck layers (blocks) with average
# pooling and ReLU as the activation function.
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.dropout = nn.Dropout(0.2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully-connected layer below that has the number of classes
        # as the output units.
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet18(num_classes=1):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)



# Train

import torch
import h5py
from torch.utils.data import random_split
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def preprocess(data_file, labels_file):
    
    # Create the custom dataset
    custom_dataset = CustomDataset(data_file, labels_file)

    # Define the sizes of the training and validation sets of 80/20 splits
    total_size = len(custom_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    # Use random_split to split the dataset
    train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])

    # Create DataLoader instances for training and validation
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader


def train(data_file, labels_file):
  # Device configuration:
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  train_loader, val_loader = preprocess(data_file, labels_file)

  model = resnet18(num_classes = 9).to(device)

  # Hyperparameters:
  num_epochs = 50
  learning_rate = 0.0001

  num_input_channels = 3
  criterion = nn.CrossEntropyLoss() # For multi-class classification.
  optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

  n_total_steps = len(train_loader)
  min_loss = 1000
  count = 0
  patience = 10
  train_losses = []
  val_losses = []

  for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

      images = images.to(device)
      images = images.to(torch.float)
      images = images.permute(0, 3, 1, 2)
      labels = labels.to(device)
      labels = labels.to(torch.long)

      # Forward pass:
      outputs = model(images)
      _, predicted = torch.max(outputs, 1)
      loss = criterion(outputs, labels) # Labels --> target.

      # Backpropagation and optimization:
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()



    # checking for early stopping
    if loss.item() < min_loss:
      min_loss = loss.item()
      count = 0
      # Save the model checkpoint
      torch.save(model, 'best_model.pth')
      # Save the model parameters to an HDF5 file during training
      with h5py.File('best_model.h5', 'w') as hf:
          for name, param in model.named_parameters():
              # Use name.replace('.', '/') to convert parameter names to valid HDF5 keys
              hf.create_dataset(name.replace('.', '/'), data=param.detach().cpu().numpy())
    else:
      count += 1
      if count == patience:
        break
    


    print(f"Epoch: {epoch + 1} / {num_epochs}, Step: {i + 1} / {n_total_steps}, Loss: {loss.item():.4f}")

  print("Finished training.")

  return model, train_loader, val_loader


def main():
  # Train the model
  data_file = 'data/augmented_data.npy'
  labels_file = 'data/augmented_labels.npy'
  model, train_loader, val_loader = train(data_file, labels_file)
  # Do something with the model as necessary


if __name__ == '__main__':
  main()
