################################################################################
# MIT License
# 
# Copyright (c) 2018
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader

import sys
sys.path.append("..")
from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter
#torch.manual_seed(42)
#np.random.seed(42)

################################################################################

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    accuracies_test = []
    accuracies_train = []
    input_length = config.input_length

    for input_length in config.input_length:
        avg_acc_train = []
        # Initialize the model that we are going to use
        if config.model_type == "RNN":
            model = VanillaRNN(input_length, config.input_dim, config.num_hidden, config.num_classes,
                               config.batch_size, config.device).to(device)

        elif config.model_type == "LSTM":
            model = LSTM(input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size,
                         config.device).to(device)

        # Initialize the dataset and data loader (note the +1)
        dataset = PalindromeDataset(input_length+1)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

        # Setup the loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), config.learning_rate)

        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            # Only for time measurement of step through network
            t1 = time.time()
            # Add more code here ...
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            optimizer.zero_grad()
            pred = model(batch_inputs)
            # print(pred.shape)
            loss = criterion(pred, batch_targets)
            loss.backward()
            optimizer.step()

            # if step == 0:
            #     print(batch_inputs.shape)
            #     print(batch_targets.shape)
            #     print(pred.shape)
            ############################################################################
            # QUESTION: what happens here and why?
            # ANSWER: it makes sure the gradients are not too high (thus preventing exploding gradients)
            ############################################################################
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
            ############################################################################

            # Add more code here ...
            loss = loss.item()

            max, pred_classes = pred.max(1)
            correct_pred = pred_classes == batch_targets
            accuracy = torch.sum(correct_pred).item()/len(correct_pred)

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if step % 10 == 0:

                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                        config.train_steps, config.batch_size, examples_per_second,
                        accuracy, loss
                ))

            # Save accuracies from last 10 steps
            if (config.train_steps - 10) < step:
                avg_acc_train.append(accuracy)


            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                mean_acc = sum(avg_acc_train)/len(avg_acc_train)
                accuracies_train.append(mean_acc)
                break

        print('Done training.')

        print("Mean last steps train accuracy: " + str(mean_acc))
        # Make test set and evaluate accuracy
        testset = PalindromeDataset(input_length + 1)
        test_loader = DataLoader(testset, 128, num_workers=1)

        avg_acc_test = []
        for step2, (batch_inputs, batch_targets) in enumerate(test_loader):
            model.c_init = nn.Parameter(torch.zeros(128, 128))
            pred = model(batch_inputs)
            # loss = criterion(pred, batch_targets)
            max, pred_classes = pred.max(1)
            correct_pred = pred_classes == batch_targets
            accuracy = torch.sum(correct_pred).item() / len(correct_pred)
            avg_acc_test.append(accuracy)
            if step2 == 78:
                break
        mean_test_acc = sum(avg_acc_test)/len(avg_acc_test)
        print("Test accuracy: " + str(accuracy))
        accuracies_test.append(mean_test_acc)


    plt.figure(1)
    plt.rcParams['font.size'] = 15
    plt.plot(config.input_length, accuracies_test, label="Accuracies test set")
    plt.plot(config.input_length, accuracies_train, label="Accuracies train set")
    plt.xlabel("Palindrome length")
    plt.ylabel("Accuracy")
    plt.title("Accuracies RNN")
    plt.legend()
    plt.savefig("Accuracies_RNN.png")
    plt.show()
 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=[20], help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=2000, help='Number of training steps')  # 10000
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")  # cuda:0

    config = parser.parse_args()

    # Train the model
    train(config)