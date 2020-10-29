"""
This module implements various modules of the network.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
        """

        self.params = {'weight': np.random.normal(0, 0.0001, (out_features, in_features)),
                       'bias': np.zeros((out_features, 1))}
        self.grads = {'weight': np.zeros((out_features, in_features)), 'bias': np.zeros((out_features, 1))}

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module
        """

        print("weight: " + str(self.params['weight'].shape))
        print("bias: " + str(self.params['bias'].shape))
        print("x: " + str(x.shape))
        self.out = (self.params['weight'] @ x.T + self.params['bias']).T
        self.x = x
        print("out: " + str(self.out.shape))
        return self.out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
    """

        self.grads['weight'] = (self.x.T @ dout).T
        self.grads['bias'] = dout
        dx = dout @ self.params['weight']
        return dx


class ReLUModule(object):
    """
    ReLU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module
        """
        self.out = np.maximum(x, 0)
        return self.out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """
        dx = dout * (self.out > 0)
        return dx


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
        """
        shift = x.max(axis=1, keepdims=True)
        y = np.exp(x - shift)
        self.out = y / y.sum(axis=1, keepdims=True)
        return self.out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        """
        # First calculate the diagonal matrix
        diagonal_x = np.eye(self.out.shape[1]) * self.out[:, np.newaxis, :]
        # Then calculate dx/dx_tilde
        dx_dx_tilde = diagonal_x - np.einsum('ij, ik -> ijk', self.out, self.out)
        # Finally calculate dx
        dx = np.einsum('ij, ijk -> ik', dout, dx_dx_tilde)
        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.

        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        """

        # This takes the argmax
        out = -np.log(x[y.astype(bool)]).mean()
        return out

    def backward(self, x, y):
        """
    Backward pass.
    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    """
        dx = (-y / x) / x.shape[0]
        return dx
