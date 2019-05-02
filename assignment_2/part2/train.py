# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import os
import time
from datetime import datetime
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
sys.path.append("..")
from part2.dataset import TextDataset
from part2.model import TextGenerationModel
# from tensorboardX import SummaryWriter

################################################################################

def train(config):

    def convert_to_right_format_batch(batch_inputs, batch_targets):
        # Convert to one-hot encoding
        batch_inputs = torch.stack(batch_inputs)
        # embedding = nn.Embedding(dataset.vocab_size, config.lstn_num_hidden)
        # embedding(batch_inputs)
        identity = torch.eye(dataset.vocab_size)
        batch_inputs = identity[batch_inputs]
        batch_targets = torch.stack(batch_targets)
        return batch_inputs, batch_targets

    def convert_sample_to_sentence(sample_input, sample_output):
        sample_input = torch.stack(sample_input)
        input_data = sample_input[:, 0]
        input_data = input_data.tolist()
        sample_output = torch.stack(sample_output)
        output = sample_output[:, 0]
        output = output.tolist()
        input_data = dataset.convert_to_string(input_data)
        output = dataset.convert_to_string(output)
        return input_data, output

    def print_sequence_to_sequence_prediction():
        testset = TextDataset(config.txt_file, 30)
        test_loader = DataLoader(testset, 1, num_workers=1)
        for step2, (test_inputs, test_targets) in enumerate(test_loader):
            input_sentence, output_sentence = convert_sample_to_sentence(test_inputs, test_targets)
            print(input_sentence)
            print(output_sentence)
            test_inputs, test_targets = convert_to_right_format_batch(test_inputs, test_targets)
            pred = model(test_inputs)
            pred = pred.view(-1, dataset.vocab_size)
            max, pred_classes = pred.max(1)
            pred_classes = pred_classes.tolist()
            pred_classes = dataset.convert_to_string(pred_classes)
            print(pred_classes)
            if step2 == 0:
                break

    # Initialize the device which to run the model on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the dataset and data loader
    dataset = TextDataset(config.txt_file, 30)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size, config.lstm_num_hidden,
                                config.lstm_num_layers, device)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), config.learning_rate)
    offset = 0

    # Loading model if available
    # load_model = False
    # if load_model:
    #     checkpoint = torch.load("model1000.pt")
    #     model.load_state_dict(checkpoint["model_state"])
    #     optimizer.load_state_dict((checkpoint["optimizer_state"]))
    #     dataset = checkpoint["dataset"]
    #     data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
    #     offset = checkpoint["offset"]

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        #######################################################
        # Convert to real step size (because training is resumed otherwise 0)
        step = step + offset

        # Convert to one-hot encoding
        batch_inputs, batch_targets = convert_to_right_format_batch(batch_inputs, batch_targets)
        optimizer.zero_grad()
        batch_size = batch_inputs.shape[1]
        h = torch.zeros(config.lstm_num_layers, batch_size, config.lstm_num_hidden)
        c = torch.zeros(config.lstm_num_layers, batch_size, config.lstm_num_hidden)
        pred, _, _ = model(batch_inputs, h, c)
        pred = pred.view(-1, dataset.vocab_size)
        batch_targets = batch_targets.view(-1)
        # print(pred.shape)

        loss = criterion(pred, batch_targets)
        loss.backward()
        optimizer.step()
        #######################################################

        loss = loss.item()
        max, pred_classes = pred.max(1)
        correct_pred = pred_classes == batch_targets
        accuracy = torch.sum(correct_pred).item() / len(correct_pred)

        # Just for time measurement
        t2 = time.time()
        examples_per_second = batch_size/float(t2-t1)
        train_steps = int(config.train_steps)
        if step % config.print_every == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    train_steps, config.batch_size, examples_per_second,
                    accuracy, loss))

        if step % config.sample_every == 0:
            # Generate some sentences by sampling from the model
            # print_sequence_to_sequence_prediction()
            h = torch.zeros(config.lstm_num_layers, 1, config.lstm_num_hidden)
            c = torch.zeros(config.lstm_num_layers, 1, config.lstm_num_hidden)
            T = None  # Set the temperature
            softmax = nn.Softmax(dim=1)
            rnd_char = random.choice(list(dataset._ix_to_char))
            pred = torch.zeros(1, 1, dataset.vocab_size)
            pred[0][0][rnd_char] = 1
            predictions = [rnd_char]
            for i in range(config.seq_length):
                pred, h, c = model(pred, h, c)
                out = pred.view(-1, dataset.vocab_size)
                if T is not None:
                    prob_dis = softmax(out / T)
                    pred_class = torch.multinomial(prob_dis, 1)
                else:
                    max, pred_class = out.max(1)
                predictions.append(pred_class.item())
                pred = torch.zeros(1, 1, dataset.vocab_size)
                pred[0][0][pred_class.item()] = 1
            predictions = dataset.convert_to_string(predictions)
            print(predictions)

        # Save model every ... steps
        if step % config.save_every == 0:
            filename = "rationality_model/model" + str(step) + ".pt"
            torch.save({
                        "dataset": dataset,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "accuracy": accuracy,
                        "loss": loss,
                        "offset": step}, filename)
            pass

        if step == train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=1000, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=1000, help='How often to sample from the model')

    # Extra params
    parser.add_argument('--save_every', type=int, default=1000000, help='How often to save the model')

    config = parser.parse_args()

    # Train the model
    train(config)
