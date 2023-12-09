# -*- coding: utf-8 -*-
"""test.ipynb
# Final Test Script
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data_file, labels_file, transform=None):
        self.data = np.load(data_file)
        self.labels = np.load(labels_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # image = self.data[idx]
        image = self.data[:, idx].reshape(300, 300, 3)  # Reshape the individual image to its original shape
        # image = self.data[:, idx].reshape(100, 100, 3)  # Reshape the individual image to its original shape
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)


        return image, label


import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

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


def test(test_data_file, test_labels_file):
    custom_test_dataset = CustomDataset(test_data_file, test_labels_file)
    # Create a DataLoader
    batch_size = 64
    shuffle = False  # Set to True if you want to shuffle the data

    test_loader = DataLoader(dataset=custom_test_dataset, batch_size=batch_size, shuffle=shuffle)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate the model
    # model = resnet18(num_classes = 9).to(device)

    model_path = 'Best_Model/best_model.pth'

    # Load the pre-trained weights
    model = torch.load(model_path)

    model.to(device)

    # Set the model to eval mode
    model.eval()

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(9)]
        n_class_samples = [0 for i in range(9)]

        i = 0
        for images, labels in test_loader:
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            images = images.permute(0, 3, 1, 2)  # Permute to [batch_size, 3, 300, 300]
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)


            probabilities = torch.nn.functional.softmax(outputs, dim=0)

            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                if label == pred:
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples

        class_acc = []
        for i in range(9):
            class_acc.append(100.0 * n_class_correct[i] / n_class_samples[i])

    print(f"Total Accuracy: {acc}%")
    for i in range(9):
        print(f"Accuracy for class {i}: {class_acc[i]}%")


def main():
    test_data_file = sys.argv[1]
    test_labels_file = sys.argv[2]
    test(test_data_file, test_labels_file)


if __name__ == "__main__":
    main()