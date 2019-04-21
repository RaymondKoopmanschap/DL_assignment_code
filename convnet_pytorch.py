"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(ConvNet, self).__init__()

    layers = []
    layers.extend(self.conv_batch_relu(n_channels, 64, 3, 1, 1))
    layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    layers.extend(self.conv_batch_relu(64, 128, 3, 1, 1))
    layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    layers.extend(self.conv_batch_relu(128, 256, 3, 1, 1))
    layers.extend(self.conv_batch_relu(256, 256, 3, 1, 1))
    layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    layers.extend(self.conv_batch_relu(256, 512, 3, 1, 1))
    layers.extend(self.conv_batch_relu(512, 512, 3, 1, 1))
    layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    layers.extend(self.conv_batch_relu(512, 512, 3, 1, 1))
    layers.extend(self.conv_batch_relu(512, 512, 3, 1, 1))
    layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    layers.append(nn.AvgPool2d(kernel_size=1, stride=1, padding=0))

    self.lin = nn.Linear(512, n_classes)

    self.layers = nn.Sequential(*layers)

  # specify input, output, kernel_size, stride and padding for each conv, batch, relu combi
  def conv_batch_relu(self, input, output, kernel_size, stride, padding):
    conv = nn.Conv2d(input, output, kernel_size, stride, padding)
    norm = nn.BatchNorm2d(output)
    relu = nn.ReLU()
    return [conv, norm, relu]
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    x = self.layers(x)
    x = x.view(x.shape[0], -1)
    x = self.lin(x)
    return x
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
