"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        """

        super(MLP, self).__init__()
        layer_dim = [n_inputs, *n_hidden, n_classes]
        layers = []
        # Initialize first linear layer
        layers.append(nn.Linear(layer_dim[0], layer_dim[1]))
        # Initialize hidden layers in the form ReLU, Linear
        for index in range(len(n_hidden)):
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(layer_dim[index + 1]))
            layers.append(nn.Linear(layer_dim[index + 1], layer_dim[index + 2]))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        """
        out = self.layers(x)
        return out
