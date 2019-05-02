import torch
import sys
sys.path.append("..")
from part2.model import TextGenerationModel
import torch.nn as nn
import torch.optim as optim
import argparse
import random

def generation(config):
    model = TextGenerationModel(64, 30, 148, 128,
                                    2, "cpu")
    optimizer = optim.Adam(model.parameters(), 1e-2)

    load_model = "./rationality_model/model" + str(config.model_steps) + ".pt"
    print("Loaded model from " + str(config.model_steps) + " steps")
    checkpoint = torch.load(load_model)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict((checkpoint["optimizer_state"]))
    dataset = checkpoint["dataset"]


    if config.text_completion:
        string = config.text_completion
        string = list(string)
        numbers = [dataset._char_to_ix[char] for char in string]

        h = torch.zeros(2, 1, 128)
        c = torch.zeros(2, 1, 128)
        T = config.temperature  # Set the temperature
        softmax = nn.Softmax(dim=1)
        predictions = numbers
        #
        last_number = numbers[-1]
        for count, number in enumerate(numbers):
            pred = torch.zeros(1, 1, dataset.vocab_size)
            pred[0][0][number] = 1
            pred, h, c = model(pred, h, c)
            out = pred.view(-1, dataset.vocab_size)
            if count == (len(numbers) - 1):
                if T is not None:
                    prob_dis = softmax(out / T)
                    pred_class = torch.multinomial(prob_dis, 1)
                else:
                    max, pred_class = out.max(1)
                predictions.append(pred_class.item())
                pred_final = torch.zeros(1, 1, dataset.vocab_size)
                pred_final[0][0][pred_class.item()] = 1
                break

        pred = pred_final

        #Now generate new sentences
        for i in range(config.length_pred):
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
        print("Completion prediction: " + predictions)

    if config.length_pred:
        # Generate some sentences by sampling from the model
        h = torch.zeros(2, 1, 128)
        c = torch.zeros(2, 1, 128)
        T = config.temperature  # Set the temperature

        softmax = nn.Softmax(dim=1)
        rnd_char = random.choice(list(dataset._ix_to_char))
        pred = torch.zeros(1, 1, dataset.vocab_size)
        pred[0][0][rnd_char] = 1
        predictions = [rnd_char]

        for i in range(config.length_pred):
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
        print("Random character prediction: " + predictions)



if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_steps', type=str, default=45000, help='Define which model you want to load by saying '
                                                                       'until how many steps it needs to be trained on')
    parser.add_argument('--length_pred', type=int, help='Define the length of the sequence from random character '
                                                        'prediction')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature of model')
    parser.add_argument('--text_completion', type=str, help='Type a text to complete')

    config = parser.parse_args()

    generation(config)
