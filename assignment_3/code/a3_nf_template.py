import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from datasets.mnist import mnist
import os
import math
import time
from torchvision.utils import make_grid


def log_prior(x):
    """
    Compute the elementwise log probability of a standard Gaussian, i.e.
    N(x | mu=0, sigma=1).
    """
    logp = torch.sum((-x ** 2) / 2 - math.log(math.sqrt(2 * math.pi)), dim=1)

    return logp


def sample_prior(size):
    """
    Sample from a standard Gaussian.
    """
    sample = torch.randn(size)

    if torch.cuda.is_available():
        sample = sample.cuda()

    return sample


def get_mask():
    mask = np.zeros((28, 28), dtype='float32')
    for i in range(28):
        for j in range(28):
            if (i + j) % 2 == 0:
                mask[i, j] = 1

    mask = mask.reshape(1, 28*28)
    mask = torch.from_numpy(mask)

    return mask


class Coupling(torch.nn.Module):
    def __init__(self, c_in, mask, n_hidden=1024):
        super().__init__()
        self.n_hidden = n_hidden

        # Assigns mask to self.mask and creates reference for pytorch.
        self.register_buffer('mask', mask)

        # Create shared architecture to generate both the translation and
        # scale variables.
        # Suggestion: Linear ReLU Linear ReLU Linear.
        self.nn = torch.nn.Sequential(
            nn.Linear(c_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU()
            )

        # Separate layers for transform and scale
        self.trans = nn.Linear(n_hidden, c_in)
        self.scale = nn.Linear(n_hidden, c_in)
        self.tanh = nn.Tanh()

        # The nn should be initialized such that the weights of the last layer
        # is zero, so that its initial transform is identity.
        self.trans.weight.data.zero_()
        self.trans.bias.data.zero_()
        self.scale.weight.data.zero_()
        self.scale.bias.data.zero_()

    def forward(self, z, ldj, reverse=False):
        # Implement the forward and inverse for an affine coupling layer. Split
        # the input using the mask in self.mask. Transform one part with
        # Make sure to account for the log Jacobian determinant (ldj).
        # For reference, check: Density estimation using RealNVP.

        split = z * self.mask

        both = self.nn(split)
        trans = self.trans(both)
        scale = self.scale(both)
        log_scale = self.tanh(scale)

        # Using equation 9 in the RealNVP paper. (Where b is self.mask and x is z)
        if not reverse:
            y = split + (1 - self.mask) * (z * torch.exp(log_scale) + trans)
            ldj = ldj + torch.sum((1 - self.mask) * log_scale, dim=1)
        else:
            y = split + (1 - self.mask) * (z - trans) * torch.exp(-log_scale)
            ldj = torch.zeros_like(ldj)

        return y, ldj


class Flow(nn.Module):
    def __init__(self, shape, n_flows=4):
        super().__init__()
        channels, = shape

        mask = get_mask()

        self.layers = torch.nn.ModuleList()

        for i in range(n_flows):
            self.layers.append(Coupling(c_in=channels, mask=mask))
            self.layers.append(Coupling(c_in=channels, mask=1-mask))

        self.z_shape = (channels,)

    def forward(self, z, logdet, reverse=False):
        if not reverse:
            for layer in self.layers:
                z, logdet = layer(z, logdet)
        else:
            for layer in reversed(self.layers):
                z, logdet = layer(z, logdet, reverse=True)

        return z, logdet


class Model(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.flow = Flow(shape)

    def dequantize(self, z):
        return z + torch.rand_like(z)

    def logit_normalize(self, z, logdet, reverse=False):
        """
        Inverse sigmoid normalization.
        """
        alpha = 1e-5

        if not reverse:
            z = z / 256.
            logdet -= np.log(256) * np.prod(z.size()[1:])

            # Logit normalize
            z = z*(1-alpha) + alpha*0.5
            logdet += torch.sum(-torch.log(z) - torch.log(1-z), dim=1)
            z = torch.log(z) - torch.log(1-z)

        else:
            # Inverse normalize
            logdet += torch.sum(torch.log(z) + torch.log(1-z), dim=1)
            z = torch.sigmoid(z)

            z = z * 256.
            logdet += np.log(256) * np.prod(z.size()[1:])

        return z, logdet

    def forward(self, input):
        """
        Given input, encode the input to z space. Also keep track of ldj.
        """
        z = input
        ldj = torch.zeros(z.size(0), device=z.device)

        z = self.dequantize(z)
        z, ldj = self.logit_normalize(z, ldj)

        z, ldj = self.flow(z, ldj)

        # Compute log_pz and log_px per example
        log_pz = log_prior(z)
        log_px = log_pz + ldj

        return log_px

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Sample from prior and create ldj.
        Then invert the flow and invert the logit_normalize.
        """
        z = sample_prior((n_samples,) + self.flow.z_shape)
        ldj = torch.zeros(z.size(0), device=z.device)

        z, ldj = self.flow(z, ldj, reverse=True)
        z, ldj = self.logit_normalize(z, ldj, reverse=True)

        z = z.view(n_samples, 1, 28, 28)

        return z


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average bpd ("bits per dimension" which is the negative
    log_2 likelihood per dimension) averaged over the complete epoch.
    """
    bpd = []
    if model.training:
        model.train()
        for imgs, _ in data:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            log_px = model(imgs)
            loss = -log_px.mean()
            bpd.append(loss.item())
            loss.backward()
            optimizer.step()

    else:
        model.eval()
        with torch.no_grad():
            for imgs, _ in data:
                imgs = imgs.to(device)
                log_px = model(imgs)
                loss = -log_px.mean()
                bpd.append(loss.item())

    avg_bpd = sum(bpd)/(len(bpd) * imgs.shape[1] * math.log(2))

    return avg_bpd


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average bpd for each.
    """
    traindata, valdata = data

    model.train()
    train_bpd = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_bpd = epoch_iter(model, valdata, optimizer)
    return train_bpd, val_bpd


def save_bpd_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train bpd')
    plt.plot(val_curve, label='validation bpd')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('bpd')
    plt.tight_layout()
    plt.savefig(filename)


def main():

    data = mnist()[:2]  # ignore test split

    model = Model(shape=[784]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs('images_nfs', exist_ok=True)

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        start = time.time()
        bpds = run_epoch(model, data, optimizer)
        train_bpd, val_bpd = bpds
        train_curve.append(train_bpd)
        val_curve.append(val_bpd)
        end = time.time()
        time_epoch = end - start
        print("[Epoch {epoch}] train bpd: {train_bpd} val_bpd: {val_bpd} time: {time_epoch}".format(
            epoch=epoch, train_bpd=train_bpd, val_bpd=val_bpd, time_epoch=time_epoch))

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functionality that is already imported.
        #  Save grid to images_nfs/
        # --------------------------------------------------------------------
        if (epoch + 1) % 5 == 0:
            imgs = model.sample(25).cpu().detach()
            grid = make_grid(imgs, nrow=5, normalize=True).permute(1, 2, 0)
            plt.imshow(grid)
            plt.savefig('images_nfs/{}.png'.format(epoch + 1))

    save_bpd_plot(train_curve, val_curve, 'nfs_bpd.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1, type=int,
                        help='max number of epochs')

    ARGS = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main()
