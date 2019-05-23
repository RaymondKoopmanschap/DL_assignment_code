import argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from scipy.stats import norm

from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        self.linear1 = nn.Linear(784, hidden_dim)
        self.tanh = nn.ReLU()
        self.log_var = nn.Linear(hidden_dim, z_dim)
        self.mean = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        h = self.linear1(input)
        log_var = self.log_var(h)
        mean = self.mean(h)
        std = torch.sqrt(torch.exp(log_var))

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.linear1 = nn.Linear(z_dim, hidden_dim)
        self.tanh = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, 784)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        """
        Perform forward pass of decoder.

        Returns mean with shape [batch_size, 784].
        """
        h = self.linear1(input)
        h = self.tanh(h)
        mean = self.linear2(h)
        mean = self.sigmoid(mean)

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)
        self.loss = nn.BCELoss()  # Does the binary cross entropy loss

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """

        batch_size = input.shape[0]
        input = input.view(batch_size, -1)
        input_size = input.shape[1]
        mean, std = self.encoder(input)

        sample = torch.randn(batch_size, self.z_dim)
        rep_sample = mean + std * sample
        means_output = self.decoder(rep_sample)
        loss_recon = self.loss(means_output, input) * input_size   # Normal BCE takes also average of pixels, but we need sum
        # This is the loss formula described in Kingma and Welling Appendix B. Only added extra minus sign
        loss_reg_avg = 0.5 * torch.sum(-1 - torch.log(std ** 2) + mean ** 2 + std ** 2) / batch_size

        average_negative_elbo = loss_reg_avg + loss_recon
        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        samples = torch.randn(n_samples, self.z_dim)
        im_means = self.decoder(samples)

        sampled_ims = torch.bernoulli(im_means)

        sampled_ims = sampled_ims.view(n_samples, 1, 28, 28)
        im_means = im_means.view(n_samples, 1, 28, 28)
        return sampled_ims, im_means.detach()


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    avg_elbo = []
    if model.training:
        model.train()
        for imgs in data:
            optimizer.zero_grad()
            elbo = model(imgs)
            avg_elbo.append(elbo)
            elbo.backward()
            optimizer.step()
    else:
        model.eval()
        with torch.no_grad():
            for imgs in data:
                elbo = model(imgs)
                avg_elbo.append(elbo)

    average_epoch_elbo = sum(avg_elbo)/len(avg_elbo)

    return average_epoch_elbo.item()


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    torch.manual_seed(42)
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())


    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functionality that is already imported.
        # --------------------------------------------------------------------
        if (epoch + 1) % 5 == 0:
            sampled_ims, im_means = model.sample(25)
            grid = make_grid(im_means, nrow=5).permute(1, 2, 0)
            plt.imshow(grid)
            plt.savefig('images_vae/{}_means.png'.format(epoch + 1))

    # --------------------------------------------------------------------
    #  Add functionality to plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    # I took this code from https://www.quora.com/How-can-I-draw-a-manifold-from-a-variational-autoencoder-in-Keras
    if ARGS.zdim == 2:
        nx = ny = 20
        x_axis = np.linspace(.05, .95, nx)
        y_axis = np.linspace(.05, .95, ny)

        grid = np.empty((28 * ny, 28 * nx))
        for i, yi in enumerate(x_axis):
            for j, xi in enumerate(y_axis):
                z = torch.from_numpy(np.array([[norm.ppf(xi), norm.ppf(yi)]]).astype('float32'))
                mean = model.decoder(z).detach()
                grid[(nx - i - 1) * 28:(nx - i) * 28, j * 28:(j + 1) * 28] = mean[0].reshape(28, 28)

        plt.imshow(grid, cmap="gray")
        plt.tight_layout()
        plt.axis('off')
        plt.savefig('manifold.png')

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
