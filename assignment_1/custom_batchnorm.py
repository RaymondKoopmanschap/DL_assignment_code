import torch
import torch.nn as nn

"""
The modules/function here implement custom versions of batch normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
"""


######################################################################################
# Code for Question 3.1
######################################################################################

class CustomBatchNormAutograd(nn.Module):
    """
    This nn.module implements a custom version of the batch norm operation for MLPs.
    The operations called in self.forward track the history if the input tensors have the
    flag requires_grad set to True.
    """

    def __init__(self, n_neurons, eps=1e-5):
        """
        Initializes CustomBatchNormAutograd object.
    
        Args:
          n_neurons: int specifying the number of neurons
          eps: small float to be added to the variance for stability

        """
        super(CustomBatchNormAutograd, self).__init__()
        self.n_neurons = n_neurons
        self.eps = eps
        self.beta = nn.Parameter(torch.zeros(self.n_neurons))
        self.gamma = nn.Parameter(torch.ones(self.n_neurons))

    def forward(self, input):
        """
        Compute the batch normalization

        Args:
          input: input tensor of shape (n_batch, n_neurons)
        Returns:
          out: batch-normalized tensor

        """
        batch_size = input.shape[0]
        assert (input.shape[1] == self.n_neurons), "Input not in the correct shape"
        mean = 1 / batch_size * torch.sum(input, dim=0)
        var = input.var(dim=0, unbiased=False)
        # Broadcasting for the win
        norm = (input - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * norm + self.beta
        return out


######################################################################################
# Code for Question 3.2 b)
######################################################################################


class CustomBatchNormManualFunction(torch.autograd.Function):
    """
    This torch.autograd.Function implements a functional custom version of the batch norm operation for MLPs.
    Using torch.autograd.Function allows you to write a custom backward function.
    The function will be called from the nn.Module CustomBatchNormManualModule
    Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
    pass is done via the backward method.
    The forward pass is not called directly but via the apply() method. This makes sure that the context objects
    are dealt with correctly. Example:
    my_bn_fct = CustomBatchNormManualFunction()
    normalized = fct.apply(input, gamma, beta, eps)
    """

    @staticmethod
    def forward(ctx, input, gamma, beta, eps=1e-5):
        """
        Compute the batch normalization

        Args:
        ctx: context object handling storing and retrival of tensors and constants and specifying
             whether tensors need gradients in backward pass
        input: input tensor of shape (n_batch, n_neurons)
        gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
        beta: mean bias tensor, applied per neuron, shpae (n_neurons)
        eps: small float added to the variance for stability
        Returns:
        out: batch-normalized tensor
        """

        batch_size = input.shape[0]
        mean = 1 / batch_size * torch.sum(input, dim=0)
        var = input.var(dim=0, unbiased=False)
        # Broadcasting for the win
        norm = (input - mean) / torch.sqrt(var + eps)
        out = gamma * norm + beta
        ctx.save_for_backward(norm, gamma, var)
        ctx.eps = eps

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
    Compute backward pass of the batch normalization.
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
    Returns:
      out: tuple containing gradients for all input arguments

    """
        normalized, gamma, var = ctx.saved_tensors
        eps = ctx.eps
        B = grad_output.shape[0]

        grad_gamma = (grad_output * normalized).sum(0)
        grad_beta = torch.sum(grad_output, dim=0)
        grad_input = torch.div(gamma, B * torch.sqrt(var + eps)) * (
                    B * grad_output - grad_beta - grad_gamma * normalized)

        # return gradients of the three tensor inputs and None for the constant eps
        return grad_input, grad_gamma, grad_beta, None


######################################################################################
# Code for Question 3.2 c)
######################################################################################

class CustomBatchNormManualModule(nn.Module):
    """
    This nn.module implements a custom version of the batch norm operation for MLPs.
    In self.forward the functional version CustomBatchNormManualFunction.forward is called.
    The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
    """

    def __init__(self, n_neurons, eps=1e-5):
        """
        Initializes CustomBatchNormManualModule object.

        Args:
          n_neurons: int specifying the number of neurons
          eps: small float to be added to the variance for stability
        """
        super(CustomBatchNormManualModule, self).__init__()
        self.n_neurons = n_neurons
        self.eps = eps
        self.beta = nn.Parameter(torch.zeros(self.n_neurons))
        self.gamma = nn.Parameter(torch.ones(self.n_neurons))

    def forward(self, input):
        """
        Compute the batch normalization via CustomBatchNormManualFunction

        Args:
          input: input tensor of shape (n_batch, n_neurons)
        Returns:
          out: batch-normalized tensor
        """
        assert input.shape[1] == self.n_neurons
        batch_norm_custom = CustomBatchNormManualFunction()
        out = batch_norm_custom.apply(input, self.gamma, self.beta, self.eps)
        return out
