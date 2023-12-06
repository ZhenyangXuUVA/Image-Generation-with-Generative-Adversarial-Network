#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from google.colab import drive
drive.mount('/content/gdrive')
import os
import shutil
import pickle


# In[6]:


get_ipython().run_line_magic('pwd', '')
get_ipython().run_line_magic('ls', '')


# In[16]:


def get_data(batch_size, image_channels, image_height_width):

    """
    Helper function to download & transform MNIST data
    """

    transform = transforms.Compose([
        transforms.Resize(image_height_width),  # reshape the MNIST images from 28x28 to 64x64
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(image_channels)], [0.5 for _ in range(image_channels)]),
    ])
    DATA_DIR = "."
    train_data = MNIST(root=DATA_DIR + 'train', train=True, transform=transform, download=True)
    val_data = MNIST(root=DATA_DIR + 'val', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)

    return train_data, train_loader, val_data, val_loader


# In[17]:


class DCGAN_Discriminator(nn.Module):
    def __init__(self, img_channels, features_disc):
        super().__init__()

        self.disc = nn.Sequential(

            # first Conv2d layer
            nn.Conv2d(img_channels, features_disc, kernel_size=4, stride=2, padding=1),       # output H x W = 32x32
            nn.LeakyReLU(0.2),

            # middle _conv2d_layer(in_channels, out_channels, kernel_size, stride, padding)
            self._conv2d_layer(features_disc, features_disc * 2, 4, 2, 1),                          # output H x W = 16x16
            self._conv2d_layer(features_disc * 2, features_disc * 4, 4, 2, 1),                      # output H x W = 8x8
            self._conv2d_layer(features_disc * 4, features_disc * 8, 4, 2, 1),                      # output H x W = 4x4

            # last conv2d layer to make 4x4 into 1x1
            nn.Conv2d(features_disc * 8, 1, kernel_size=4, stride=2, padding=0),                 # output H x W = 1x1
        )


    # helper function to abstract each Conv2d layer in the middle
    def _conv2d_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )


    def forward(self, x):
        # input: batch_size x img_channels x 64 x 64
        return self.disc(x)



class DCGAN_Generator(nn.Module):

    def __init__(self, noise_dim, img_channels, features_gen):
        super().__init__()

        self.gen = nn.Sequential(

            self._convTranspose2d_layer(noise_dim, features_gen * 16, 4, 1, 0),         # output H x W = 4x4
            self._convTranspose2d_layer(features_gen * 16, features_gen * 8, 4, 2, 1),  # output H x W = 8x8
            self._convTranspose2d_layer(features_gen * 8, features_gen * 4, 4, 2, 1),   # output H x W = 16x16
            self._convTranspose2d_layer(features_gen * 4, features_gen * 2, 4, 2, 1),   # output H x W = 32x32

            # last ConvTranspose2d layer to output image size = 64 x 64
            nn.ConvTranspose2d(
                features_gen * 2, img_channels, kernel_size=4, stride=2, padding=1      # output H x W = 64x64
            ),

            nn.Tanh(),
        )


    # helper function to abstract each ConvTranspose2d layer in the middle
    def _convTranspose2d_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        # Input: batch_size x noise_dim x 1 x 1
        return self.gen(x)


# In[18]:


def initialize_weights(model):
    # Initialize weights with mean=0, stdev=0.02 according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


# In[19]:



def train(discriminator, generator, disc_optimizer, gen_optimizer, criterion, train_loader, val_loader, epochs, device):

    """Train the GANs network"""

    # Tensorboard for logging
    writer_fake = SummaryWriter(f"logs/DCGAN_MNIST/fake")
    writer_real = SummaryWriter(f"logs/DCGAN_MNIST/real")
    log_step = 0


    for epoch in range(epochs):

        disc_losses = []
        gen_losses = []

        for batch_idx, (real_images, _) in enumerate(train_loader):
            real_images = real_images.to(device)
            batch_size = real_images.shape[0]


            #===============================
            # Discriminator Network Training
            #===============================

            # Loss of the discriminator on MNIST image inputs and real_labels
            discriminator.train()
            disc_real = discriminator(real_images)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))

            # Loss of the discriminator on fake images generated by the generator
            noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)

            generator.eval()
            with torch.no_grad():
              fake_images = generator(noise)

            disc_fake = discriminator(fake_images)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

            # Total discriminator loss
            disc_loss_total = loss_disc_real + loss_disc_fake

            # Backpropagating the discriminator loss
            disc_optimizer.zero_grad()
            disc_loss_total.backward()
            disc_optimizer.step()


            #===============================
            # Generator Network Training
            #===============================

            # Loss of the generator
            generator.train()
            noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)
            fake_images = generator(noise)

            disc_fake = discriminator(fake_images)
            loss_gen = criterion(disc_fake, torch.ones_like(disc_fake))

            # Backpropagating the generator loss
            gen_optimizer.zero_grad()
            loss_gen.backward()
            gen_optimizer.step()

            # Log the losses
            disc_losses.append(disc_loss_total.item())
            gen_losses.append(loss_gen.item())

            if batch_idx == 0:

                print(
                    f"Epoch [{epoch}/{epochs}] \
                      Learning Rate: {LEARNING_RATE}\
                      Discriminator Mean Loss: {torch.mean(torch.FloatTensor(disc_losses)):.4f}, \
                      Generator Mean Loss: {torch.mean(torch.FloatTensor(gen_losses)):.4f}"
                )

                with torch.no_grad():
                    fake_images = generator(val_noise)
                    img_grid_fake = torchvision.utils.make_grid(fake_images[:64], normalize=True)
                    img_grid_real = torchvision.utils.make_grid(real_images[:64], normalize=True)

                    writer_fake.add_image("Generated Fake MNIST Images", img_grid_fake, global_step=log_step)
                    writer_real.add_image("MNIST Real Images", img_grid_real, global_step=log_step)

                    log_step += 1

    return disc_losses, gen_losses


# In[20]:


LEARNING_RATE = 0.0002
BATCH_SIZE = 128
EPOCHS = 10
IMAGE_HEIGHT_WIDTH = 64
IMAGE_CHANNELS = 1
FEATURES_DISC = 64
FEATURES_GEN = 64
noise_dim = 100
device = "cuda" if torch.cuda.is_available() else "cpu"


# In[21]:


discriminator = DCGAN_Discriminator(IMAGE_CHANNELS, FEATURES_DISC).to(device)
generator = DCGAN_Generator(noise_dim, IMAGE_CHANNELS, FEATURES_GEN).to(device)

initialize_weights(generator)
initialize_weights(discriminator)

val_noise = torch.randn(64, noise_dim, 1, 1).to(device)


# In[22]:



gen_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCEWithLogitsLoss()


# In[23]:


train_data, train_loader, val_data, val_loader = get_data(BATCH_SIZE, IMAGE_CHANNELS, IMAGE_HEIGHT_WIDTH)


# In[24]:


train(discriminator, generator, disc_optimizer, gen_optimizer, criterion, train_loader, val_loader, EPOCHS, device)


# In[25]:


torch.save(discriminator.state_dict(), 'dcgan_discriminator_state_dict.pth')
torch.save(generator.state_dict(), 'dcgan_generator_state_dict.pth')


# In[26]:


drive.mount('/content/drive')


# In[27]:


#Display images on tensorboard

get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', "--logdir='./logs'")


# In[28]:


#Train the DCGANs network, with 100 epochs

epochs = 100
for lr in [0.0001, 0.00001, 0.000001]:
  LEARNING_RATE = lr
  generator =  DCGAN_Generator(noise_dim, IMAGE_CHANNELS, FEATURES_GEN).to(device)
  discriminator = DCGAN_Discriminator(IMAGE_CHANNELS, FEATURES_DISC).to(device)

  initialize_weights(generator)
  initialize_weights(discriminator)

  val_noise = torch.randn(64, noise_dim, 1, 1).to(device)

  gen_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
  disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
  criterion = nn.BCEWithLogitsLoss()

  train_data, train_loader, val_data, val_loader = get_data(BATCH_SIZE, IMAGE_CHANNELS, IMAGE_HEIGHT_WIDTH)

  disc_losses, gen_losses = train(discriminator, generator, disc_optimizer, gen_optimizer, criterion, train_loader, val_loader, epochs, device)

  dirname = f'/content/gdrive/MyDrive/dcgan/{lr:.0E}'

  if os.path.exists(dirname):
      shutil.rmtree(dirname)

  shutil.copytree("./logs/DCGAN_MNIST", f'{dirname}/tensorboard')

  with open(f"{dirname}/results.pkl", "wb") as f:
      pickle.dump(disc_losses, f)
      pickle.dump(gen_losses, f)

  torch.save(generator.state_dict(), f"{dirname}/generator.pth")
  torch.save(generator.state_dict(), f"{dirname}/generator.pth")

