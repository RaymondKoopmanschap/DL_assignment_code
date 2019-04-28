"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")
import argparse
from assignment_1 import cifar10_utils


from assignment_1.mlp_numpy import MLP
from assignment_1.modules import CrossEntropyModule
from assignment_1.cifar10_utils import *
from matplotlib import pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 50  # 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 10  # 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

# print(sys.path)
# cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
# print(cifar10)
# x, y = cifar10['train'].next_batch(BATCH_SIZE_DEFAULT)

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
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  pred_class = np.argmax(predictions, axis=1)
  label_class = np.argmax(targets, axis=1)
  correct = pred_class == label_class
  accuracy = np.sum(correct)/len(correct)
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################
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
  cross_entropy = CrossEntropyModule()

  # Stuff used for printing the graphs
  x_test, y_test = cifar10['test'].images, cifar10['test'].labels
  x_test = x_test.reshape(10000, -1)
  losses_train = []
  losses_test = []
  accuracies_train = []
  accuracies_test = []
  iterations = []

  # Training
  for i in range(FLAGS.max_steps):
    # Get next batch
    x, y = cifar10['train'].next_batch(batch_size)
    x = x.reshape(batch_size, -1)
    # Forward prop
    pred = network.forward(x)
    # Back prop
    dout = cross_entropy.backward(pred, y)
    network.backward(dout)
    # # Update weights
    for num_layer in range(len(network.lin_layers)):
      network.lin_layers[num_layer].params['weight'] = network.lin_layers[num_layer].params['weight'] \
                                                       - l_rate * network.lin_layers[num_layer].grads['weight']
    if i % FLAGS.eval_freq == 0 or i == FLAGS.max_steps - 1:
      loss = cross_entropy.forward(pred, y)
      print("loss: " + str(loss))

      # Stuff used for plotting
      iterations.append(i)
      loss = cross_entropy.forward(pred, y)
      losses_train.append(loss)
      pred_test = network.forward(x_test)
      loss_test = cross_entropy.forward(pred_test, y_test)
      losses_test.append(loss_test)
      acc_train = accuracy(pred, y)
      acc_test = accuracy(pred_test, y_test)
      accuracies_train.append(acc_train)
      accuracies_test.append(acc_test)

  # Evaluate test set and get final accuracy
  x, y = cifar10['test'].images, cifar10['test'].labels
  x = x.reshape(x.shape[0], -1)
  pred = network.forward(x)
  acc = accuracy(pred, y)
  print(acc)
  loss_and_acc = {}
  loss_and_acc["iterations"] = iterations
  loss_and_acc["losses_train"] = losses_train
  loss_and_acc["losses_test"] = losses_test
  loss_and_acc["acc_train"] = accuracies_train
  loss_and_acc["acc_test"] = accuracies_test
  with open("loss_and_accuracy.pickle", 'wb') as file:
    pickle.dump(loss_and_acc, file, protocol=pickle.HIGHEST_PROTOCOL)


  # Plot loss and accuracy curves
  plt.figure(1)
  plt.rcParams['font.size'] = 20
  plt.plot(iterations, losses_train, label="Loss curve train")
  plt.plot(iterations, losses_test, label="Loss curve test")
  plt.xlabel("Number of iterations")
  plt.ylabel("Loss")
  plt.title("Losses numpy")
  plt.legend()
  plt.figure(2)
  plt.plot(iterations, accuracies_train, label="Accuracy train")
  plt.plot(iterations, accuracies_test, label="Accuracy test")
  plt.xlabel("Number of iterations")
  plt.ylabel("Loss")
  plt.title("Accuracies numpy")
  plt.legend()
  plt.show()
  ########################
  # END OF YOUR CODE    #
  #######################

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
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()