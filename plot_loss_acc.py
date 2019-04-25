import pickle
import matplotlib.pyplot as plt

with open("loss_and_accuracy.pickle", 'rb') as file:
    loss_and_acc = pickle.load(file)

iterations = loss_and_acc["iterations"]
losses_train = loss_and_acc['losses_train']
losses_test = loss_and_acc['losses_test']
accuracies_train = loss_and_acc['acc_train']
accuracies_test = loss_and_acc['acc_test']

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