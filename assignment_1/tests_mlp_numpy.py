import unittest
from assignment_1.train_mlp_numpy import *
from assignment_1 import cifar10_utils
from matplotlib import pyplot as plt

np.random.seed(42)

class mlp_numpy_test(unittest.TestCase):


    def test_number_of_layers(self):
        n_inputs = 3
        n_hidden = [2, 3]
        n_classes = 2
        network = MLP(n_inputs, n_hidden, n_classes)
        self.assertEqual(len(network.lin_layers), 1 + len(n_hidden))
        self.assertEqual(len(network.relu_layers), len(n_hidden))


# Initialization
batch_size = 200
cifar10 = cifar10_utils.get_cifar10('./cifar10/cifar-10-batches-py')
x, y = cifar10['train'].next_batch(batch_size)
x = x.reshape(batch_size, -1)

x_test, y_test = cifar10['test'].images, cifar10['test'].labels
x_test = x_test.reshape(10000, -1)

n_inputs = x.shape[1]
n_hidden = [100]
n_classes = y.shape[1]
l_rate = 2e-3
losses_train = []
losses_test = []
accuracies_train = []
accuracies_test = []
iterations = []

network = MLP(n_inputs, n_hidden, n_classes)
cross_entropy = CrossEntropyModule()

for i in range(1500):
    # Training
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

    if i % 100 == 0 or i == 1499:
        iterations.append(i)
        loss = cross_entropy.forward(pred, y)
        losses_train.append(loss)
        print(loss)
        pred_test = network.forward(x_test)
        loss_test = cross_entropy.forward(pred_test, y_test)
        losses_test.append(loss_test)
        acc_train = accuracy(pred, y)
        acc_test = accuracy(pred_test, y_test)
        accuracies_train.append(acc_train)
        accuracies_test.append(acc_test)

# Get test set
x, y = cifar10['test'].images, cifar10['test'].labels
x = x.reshape(10000, -1)
pred = network.forward(x_test)
acc = accuracy(pred, y_test)
print(acc)

# Plot loss curve
plt.ylabel("Loss")
plt.rcParams['font.size'] = 20

plt.plot(iterations, losses_train, label="Loss curve train")
plt.plot(iterations, losses_test, label="Loss curve test")
plt.title("Losses numpy")
plt.show()
plt.plot(iterations, accuracies_train, label="Accuracy train")
plt.plot(iterations, accuracies_test, label="Accuracy test")
plt.xlabel("Number of iterations")
plt.title("Accuracies numpy")
plt.legend()
plt.show()