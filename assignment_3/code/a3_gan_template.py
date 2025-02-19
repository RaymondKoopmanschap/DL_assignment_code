import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import matplotlib.pyplot as plt
import time


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.linear1 = nn.Linear(args.latent_dim, 128)
        self.leakyReLU = nn.LeakyReLU(0.2)

        self.linear2 = nn.Linear(128, 256)
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.linear3 = nn.Linear(256, 512)
        self.batchnorm3 = nn.BatchNorm1d(512)

        self.linear4 = nn.Linear(512, 1024)
        self.batchnorm4 = nn.BatchNorm1d(1024)

        self.linear5 = nn.Linear(1024, 784)
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.linear1(z)
        x = self.leakyReLU(x)

        x = self.linear2(x)
        x = self.batchnorm2(x)
        x = self.leakyReLU(x)

        x = self.linear3(x)
        x = self.batchnorm3(x)
        x = self.leakyReLU(x)

        x = self.linear4(x)
        x = self.batchnorm4(x)
        x = self.leakyReLU(x)

        x = self.linear5(x)
        x = self.tanh(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.linear1 = nn.Linear(784, 512)
        self.leakyReLU = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.25)

        self.linear2 = nn.Linear(512, 256)

        self.linear3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        batch_size = img.shape[0]
        x = img.view(batch_size, -1)
        x = self.linear1(x)
        x = self.leakyReLU(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.leakyReLU(x)
        x = self.dropout(x)

        x = self.linear3(x)
        x = self.sigmoid(x)

        return x


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device):

    criterion = nn.BCELoss()

    for epoch in range(args.n_epochs):
        print("epoch: " + str(epoch))
        start = time.time()
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            imgs = imgs.to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Generator
            # ---------------
            noise = torch.randn(batch_size, args.latent_dim).to(device)
            optimizer_G.zero_grad()
            gen_imgs = generator(noise)
            gen_imgs_probs = discriminator(gen_imgs)
            loss_gen = criterion(gen_imgs_probs, real_labels)  # Tries to make real images
            loss_gen.backward()
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()
            real_imgs_probs = discriminator(imgs)
            loss_real_img = criterion(real_imgs_probs, real_labels)
            loss_real_img.backward()

            # Fake images
            gen_imgs_probs = discriminator(gen_imgs.detach()) # The generator
            loss_fake_img = criterion(gen_imgs_probs, fake_labels)
            loss_fake_img.backward()
            optimizer_D.step()

            # if i % 100 == 0:
            #     print(loss_gen.item(), loss_real_img.item(), loss_fake_img.item())
            #     acc_real = torch.sum(real_imgs_probs > 0.5).item()/real_imgs_probs.shape[0]
            #     acc_fake = torch.sum(gen_imgs_probs < 0.5).item()/gen_imgs_probs.shape[0]
            #     print("acc real:" + str(acc_real) + " acc fake:" + str(acc_fake))

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                gen_imgs = gen_imgs.view(batch_size, 1, imgs.shape[2], imgs.shape[3])
                save_image(gen_imgs[:25],
                           'images/{}.png'.format(epoch + 1),
                           nrow=5, normalize=True)

        end = time.time()
        print("time for 1 epoch:" + str(end-start))


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),
                                                (0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # Initialize models and optimizers
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "images/mnist_generator_" + args.suffix + ".pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--generate_images', action='store_true', default=False)
    parser.add_argument('--suffix', type=str, default='', help='can modify filename')
    args = parser.parse_args()

    if args.generate_images:



        generator = Generator()
        generator.load_state_dict(torch.load('mnist_generator.pt', map_location='cpu'))
        generator.eval()

        f, ax = plt.subplots(1, 9)
        noise = torch.randn(2, args.latent_dim)
        inter_noise = torch.from_numpy(np.linspace(noise[0,:], noise[1,:], 9))
        gen_imgs = generator(inter_noise)
        gen_imgs = gen_imgs.reshape((9, 28, 28)).detach()

        for i in range(9):
            ax[i].imshow(gen_imgs[i, :, :], cmap='gray')
            ax[i].axis('off')

        plt.savefig('images/interpolated.png')
        plt.show()

    else:
        main()



