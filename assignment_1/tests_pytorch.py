from assignment_1.train_mlp_pytorch import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from assignment_1 import cifar10_utils

torch.manual_seed(42)
np.random.seed(42)

class Net(nn.Module):

    def __init__(self, n_inputs, n_hidden, n_classes):
        super(Net, self).__init__()
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
        out = self.layers(x)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



# Initialization
batch_size = 200
cifar10 = cifar10_utils.get_cifar10('./cifar10/cifar-10-batches-py')
x, y = cifar10['train'].next_batch(batch_size)
x = x.reshape(batch_size, -1)
n_inputs = x.shape[1]
n_hidden = [400, 300, 200, 100]
n_classes = y.shape[1]
l_rate = 2e-3
network = Net(n_inputs, n_hidden, n_classes)
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
for i in range(2000):
    x_train, y_train = cifar10['train'].next_batch(batch_size)
    x = x_train.reshape(batch_size, -1)
    x = torch.from_numpy(x)
    y = np.argmax(y_train, axis=1)
    y = torch.from_numpy(y)

    optimizer.zero_grad()
    pred = network(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    if i % 100 == 99:
        print(loss.item())
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

# Get test set
x_test, y_test = cifar10['test'].images, cifar10['test'].labels
x_test = x_test.reshape(10000, -1)
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