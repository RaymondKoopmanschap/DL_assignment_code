"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import sys
sys.path.append("..")
from assignment_1.mlp_pytorch import MLP
from assignment_1 import cifar10_utils
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '400,300,200,100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 2000
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    """

    pred_class = np.argmax(predictions, axis=1)
    label_class = np.argmax(targets, axis=1)
    correct = pred_class == label_class
    accuracy = np.sum(correct) / len(correct)
    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.
    """

    np.random.seed(42)
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    # Set torch manual seed for reproducability
    torch.manual_seed(42)
    # Initialization
    l_rate = FLAGS.learning_rate
    batch_size = FLAGS.batch_size
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    x, y = cifar10['train'].next_batch(batch_size)
    x = x.reshape(batch_size, -1)
    n_inputs = x.shape[1]
    n_hidden = dnn_hidden_units
    n_classes = y.shape[1]
    network = MLP(n_inputs, n_hidden, n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), l_rate)

    # Stuff used for printing the graphs
    x_test_np, y_test_np = cifar10['test'].images, cifar10['test'].labels
    x_test = x_test_np.reshape(10000, -1)
    x_test = torch.from_numpy(x_test)
    y_test = np.argmax(y_test_np, axis=1)
    y_test = torch.from_numpy(y_test)
    losses_train = []
    losses_test = []
    accuracies_train = []
    accuracies_test = []
    iterations = []

    # Training
    for i in range(FLAGS.max_steps):
        # Get next batch
        x_train, y_train = cifar10['train'].next_batch(batch_size)
        x = x_train.reshape(batch_size, -1)
        x = torch.from_numpy(x)
        y = np.argmax(y_train, axis=1)
        y = torch.from_numpy(y)
        optimizer.zero_grad()
        # Forward prop
        pred = network(x)
        # Back prop
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        if i % FLAGS.eval_freq == FLAGS.eval_freq - 1:
            print("loss: " + str(loss.item()))
            # Stuff used for plotting
            iterations.append(i)
            losses_train.append(loss.item())
            pred_test = network(x_test)
            loss_test = criterion(pred_test, y_test)
            losses_test.append(loss_test)
            pred = pred.data.numpy()
            acc_train = accuracy(pred, y_train)
            pred_test = pred_test.data.numpy()
            acc_test = accuracy(pred_test, y_test_np)
            accuracies_train.append(acc_train)
            accuracies_test.append(acc_test)

    # Get test accuracy
    x_test, y_test = cifar10['test'].images, cifar10['test'].labels
    x_test = x_test.reshape(x_test.shape[0], -1)
    x_test = torch.from_numpy(x_test)
    pred = network(x_test)
    pred = pred.data.numpy()
    acc = accuracy(pred, y_test)
    print(acc)

    # Plot loss and accuracy curves
    plt.figure(1)
    plt.rcParams['font.size'] = 20
    plt.plot(iterations, losses_train, label="Loss curve train")
    plt.plot(iterations, losses_test, label="Loss curve test")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.title("Losses pytorch")
    plt.legend()
    plt.figure(2)
    plt.plot(iterations, accuracies_train, label="Accuracy train")
    plt.plot(iterations, accuracies_test, label="Accuracy test")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.title("Accuracies pytorch")
    plt.legend()
    plt.show()

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main()
