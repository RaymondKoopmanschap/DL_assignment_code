"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from assignment_1.modules import *

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
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
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # Concatenate all dimensions for easier initialization
    layer_dim = [n_inputs, *n_hidden, n_classes]
    self.lin_layers = []
    self.relu_layers = []
    # Initialize first layer
    self.lin_layers.append(LinearModule(layer_dim[0], layer_dim[1]))
    # Initialize hidden layers
    for index in range(len(n_hidden)):
        self.relu_layers.append(ReLUModule())
        self.lin_layers.append(LinearModule(layer_dim[index+1], layer_dim[index + 2]))
    # Initialize last layer
    self.softmax = SoftMaxModule()
    self.cross_entropy = CrossEntropyModule()
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
    # Forward pass of first layer
    out = self.lin_layers[0].forward(x)
    for num_layer in range(len(self.relu_layers)):
      out = self.relu_layers[num_layer].forward(out)
      out = self.lin_layers[num_layer + 1].forward(out)
    out = self.softmax.forward(out)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx = self.softmax.backward(dout)
    for num_layer in reversed(range(len(self.relu_layers))):
      dx = self.lin_layers[num_layer + 1].backward(dx)
      dx = self.relu_layers[num_layer].backward(dx)
    dx = self.lin_layers[0].backward(dx)
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx
