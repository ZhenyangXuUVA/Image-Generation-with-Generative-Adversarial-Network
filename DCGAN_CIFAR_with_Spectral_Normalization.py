#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import shutil
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy.linalg import sqrtm
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from google.colab import drive
from sklearn.metrics import accuracy_score


# In[ ]:


drive.mount('/content/gdrive')


# In[ ]:


def get_data(batch_size, image_channels, image_height_width):

    """
    Helper function to download & transform CIFAR data
    """

    transform = transforms.Compose([
        transforms.Resize(image_height_width),  # reshape the CIFAR images from 32x32 to 64x64
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(image_channels)], [0.5 for _ in range(image_channels)]),
    ])

    train_data = CIFAR10(root='./dataset', train=True, transform=transform, download=True)
    val_data = CIFAR10(root='./dataset', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)

    return train_data, train_loader, val_data, val_loader


# In[ ]:


class DCGAN_Discriminator(nn.Module):
    def __init__(self, img_channels, features_disc):
        super().__init__()

        self.disc = nn.Sequential(

            # first Conv2d layer
            spectral_norm(nn.Conv2d(img_channels, features_disc, kernel_size=4, stride=2, padding=1)),       # output H x W = 32x32
            nn.LeakyReLU(0.2, inplace=True),

            # middle _conv2d_layer(in_channels, out_channels, kernel_size, stride, padding)
            self._conv2d_layer(features_disc, features_disc * 2, 4, 2, 1),                          # output H x W = 16x16
            self._conv2d_layer(features_disc * 2, features_disc * 4, 4, 2, 1),                      # output H x W = 8x8
            self._conv2d_layer(features_disc * 4, features_disc * 8, 4, 2, 1),                      # output H x W = 4x4

            # last conv2d layer to make 4x4 into 1x1
            spectral_norm(nn.Conv2d(features_disc * 8, 1, kernel_size=4, stride=2, padding=0)),                 # output H x W = 1x1
        )


    # helper function to abstract each Conv2d layer in the middle
    def _conv2d_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            spectral_norm(nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            )),
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


# In[ ]:


def initialize_weights(model):
    # Initialize weights with mean=0, stdev=0.02 according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


# In[ ]:



def train(discriminator, generator, disc_optimizer, gen_optimizer, criterion, train_loader, val_loader, epochs, device):

    """
    Helper function to train the DCGAN network with spectral normalization
    return: mean disc_loss and gen_loss for each epoch
    """

    # Tensorboard for logging
    writer_fake = SummaryWriter(f"logs/SN_DCGAN_CIFAR/fake")
    writer_real = SummaryWriter(f"logs/SN_DCGAN_CIFAR/real")
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

            # Loss of the discriminator on CIFAR image inputs and real_labels
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



        print(
            f"Epoch [{epoch}/{epochs}] \
              Learning Rate: {LEARNING_RATE}\
              SN Discriminator Mean Loss: {torch.mean(torch.FloatTensor(disc_losses)):.4f}, \
              SN Generator Mean Loss: {torch.mean(torch.FloatTensor(gen_losses)):.4f}"
        )

        with torch.no_grad():
            fake_images = generator(val_noise)
            img_grid_fake = torchvision.utils.make_grid(fake_images[:64], normalize=True)
            img_grid_real = torchvision.utils.make_grid(real_images[:64], normalize=True)

            writer_fake.add_image("Fake CIFAR Images", img_grid_fake, global_step=log_step)
            writer_real.add_image("Real CIFAR Images", img_grid_real, global_step=log_step)

            log_step += 1

    return disc_losses, gen_losses


# In[ ]:


LEARNING_RATE = 0.0001
BATCH_SIZE = 128
EPOCHS = 25
IMAGE_HEIGHT_WIDTH = 64
IMAGE_CHANNELS = 3
FEATURES_DISC = 64
FEATURES_GEN = 64
noise_dim = 100
device = "cuda" if torch.cuda.is_available() else "cpu"


# In[ ]:


discriminator = DCGAN_Discriminator(IMAGE_CHANNELS, FEATURES_DISC).to(device)
generator = DCGAN_Generator(noise_dim, IMAGE_CHANNELS, FEATURES_GEN).to(device)

initialize_weights(generator)
initialize_weights(discriminator)

val_noise = torch.randn(64, noise_dim, 1, 1).to(device)


# In[ ]:



gen_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCEWithLogitsLoss()


# In[ ]:


train_data, train_loader, val_data, val_loader = get_data(BATCH_SIZE, IMAGE_CHANNELS, IMAGE_HEIGHT_WIDTH)


# In[ ]:


disc_losses, gen_losses = train(discriminator, generator, disc_optimizer, gen_optimizer, criterion, train_loader, val_loader, EPOCHS, device)


# In[ ]:


torch.save(discriminator.state_dict(), 'sn_dcgan_discriminator_state_dict.pth')
torch.save(generator.state_dict(), 'sn_dcgan_generator_state_dict.pth')


# In[ ]:


#Display images on tensorboard

# %load_ext tensorboard
# %tensorboard --logdir='./logs'


# In[ ]:


del train_data, train_loader, disc_losses, gen_losses, discriminator, val_noise
torch.cuda.empty_cache()


# In[ ]:


##################################
# Helper Functions for Evaluation
##################################


# Initialize the pretrained Inception v3 for computing FID score
inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device)
inception_model.fc = torch.nn.Identity()
inception_model.eval()

def extract_features(tensors, model):
    # Ensure model is in evaluation mode
    model.eval()

    # Preprocess and normalize the tensors if they're not already
    preprocess = transforms.Compose([
        transforms.Resize((299, 299), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize an empty list to hold features
    features_list = []

    # No gradients needed
    with torch.no_grad():
        for tensor in tensors:
            # Reshape and normalize the tensor
            if tensor.ndim == 3:  # If single image tensor, add batch dimension
                tensor = tensor.unsqueeze(0)
            tensor = preprocess(tensor)

            # Extract features
            feature = model(tensor)
            features_list.append(feature.cpu().numpy())

    # Concatenate all features into a single numpy array
    features = np.concatenate(features_list, axis=0)
    return features


def calculate_fid(real_features, fake_features):
    # Check for NaNs or Infs in the real features
    if np.any(np.isnan(real_features)) or np.any(np.isinf(real_features)):
        real_features = np.nan_to_num(real_features)

    # Check for NaNs or Infs in the fake features
    if np.any(np.isnan(fake_features)) or np.any(np.isinf(fake_features)):
        fake_features = np.nan_to_num(fake_features)

    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)

    eps = 1e-6
    covmean, _ = sqrtm((sigma_real + eps * np.eye(sigma_real.shape[0])) @ (sigma_fake + eps * np.eye(sigma_fake.shape[0])), disp=False)
    ssdiff = np.sum((mu_real - mu_fake) ** 2.0)

    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid_score = ssdiff + np.trace(sigma_real + sigma_fake - 2.0 * covmean)

    return fid_score


# In[ ]:


##############################################################
# Helper function to compute FID score on validation dataset
##############################################################

def get_FID_score(val_data, val_loader):
    generator.eval()
    num_images = len(val_data)  # Number of images in the validation set
    noise = torch.randn(num_images, noise_dim, 1, 1).to(device)


    #extract features for real images
    real_features_list = []
    for real_images, _ in val_loader:
        real_images = real_images.to(device)
        current_features = extract_features(real_images, inception_model)
        real_features_list.append(current_features)
        # save memory
        del real_images, current_features
        torch.cuda.empty_cache()

    ## Concatenate all feature batches
    real_features = np.concatenate(real_features_list, axis=0)

    # extract features for fake images
    fake_features_list = []

    with torch.no_grad():
        for _ in range(0, num_images, BATCH_SIZE):
            noise_batch = torch.randn(BATCH_SIZE, noise_dim, 1, 1).to(device)
            fake_images_batch = generator(noise_batch)
            current_features = extract_features(fake_images_batch, inception_model)
            fake_features_list.append(current_features)
            # save memory
            del noise_batch, fake_images_batch, current_features
            torch.cuda.empty_cache()

    # Concatenate all feature batches
    fake_features = np.concatenate(fake_features_list, axis=0)

    nan_count_real = np.isnan(real_features).sum()
    inf_count_real = np.isinf(real_features).sum()

    nan_count_fake = np.isnan(fake_features).sum()
    inf_count_fake = np.isinf(fake_features).sum()

    print("Real features - NaNs:", nan_count_real, "Infs:", inf_count_real)
    print("Fake features - NaNs:", nan_count_fake, "Infs:", inf_count_fake)

    # compute FID score
    SN_DCGAN_FID_score = calculate_fid(real_features, fake_features)

    # save memory
    del real_features, real_features_list, fake_features, fake_features_list
    torch.cuda.empty_cache()

    return SN_DCGAN_FID_score


# In[ ]:


# get FID score
SN_DCGAN_FID_score = get_FID_score(val_data, val_loader)
print('SN_DCGAN_FID_score = ', SN_DCGAN_FID_score)


# In[ ]:


##############################################################
# Helper function to evaluate discriminator accuracy
##############################################################

def evaluate_discriminator(discriminator, val_data, val_loader):
    discriminator.eval()

    # Get real images
    real_images = []
    for images, _ in val_loader:
        images = images.to(device)
        real_images.append(images)
    real_images = torch.cat(real_images, 0)


    # Get fake images
    fake_images = []
    num_fake_images = len(real_images)

    for _ in range(0, num_fake_images, BATCH_SIZE):
        noise = torch.randn(BATCH_SIZE, noise_dim, 1, 1).to(device)
        batch_fake_images = generator(noise).to(device)
        fake_images.append(batch_fake_images)
    fake_images = torch.cat(fake_images, 0)


    # Concatenate real and fake images
    all_images = torch.cat((real_images, fake_images), 0)

    # Labels: 1 for real images, 0 for fake images
    real_labels = torch.ones(real_images.size(0)).to(device)
    fake_labels = torch.zeros(fake_images.size(0)).to(device)
    all_labels = torch.cat((real_labels, fake_labels), 0)

    # Predictions
    with torch.no_grad():
        predictions = discriminator(all_images).view(-1)

    # Convert predictions to binary (0 or 1)
    predictions_binary = torch.round(torch.sigmoid(predictions))

    # Calculate accuracy
    accuracy = accuracy_score(all_labels.cpu(), predictions_binary.cpu())

    return accuracy


# In[ ]:





# In[ ]:


get_ipython().system('ls /content/gdrive/MyDrive/dcgan_sn')


# In[ ]:


#Train the DCGAN_CIFAR_with_Spectral_Normalization, with 100 epochs

epochs = 80
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


    # get the FID score
    SN_DCGAN_FID_score = get_FID_score(val_data, val_loader)
    print(f'learning rate = {lr},           SN_DCGAN_FID_score = {SN_DCGAN_FID_score}')

    dirname = f'/content/gdrive/MyDrive/dcgan_sn/{lr:.0E}'

    if os.path.exists(dirname):
        shutil.rmtree(dirname)

    shutil.copytree("./logs/SN_DCGAN_CIFAR", f'{dirname}/tensorboard')

    with open(f"{dirname}/SN_FID.pkl", "wb") as f:
        pickle.dump(SN_DCGAN_FID_score, f)

    with open(f"{dirname}/sn_results.pkl", "wb") as f:
        pickle.dump(disc_losses, f)
        pickle.dump(gen_losses, f)

    torch.save(generator.state_dict(), f"{dirname}/sn_generator.pth")
    torch.save(discriminator.state_dict(), f"{dirname}/sn_discriminator.pth")


# In[ ]:


#Train the DCGAN_CIFAR_with_Spectral_Normalization, with 100 epochs

epochs = 100
for lr in [0.00001, 0.000001]:
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


    # get the FID score
    SN_DCGAN_FID_score = get_FID_score(val_data, val_loader)
    print(f'learning rate = {lr},           SN_DCGAN_FID_score = {SN_DCGAN_FID_score}')

    dirname = f'/content/gdrive/MyDrive/dcgan_sn/{lr:.0E}'

    if os.path.exists(dirname):
        shutil.rmtree(dirname)

    shutil.copytree("./logs/SN_DCGAN_CIFAR", f'{dirname}/tensorboard')

    with open(f"{dirname}/SN_FID.pkl", "wb") as f:
        pickle.dump(SN_DCGAN_FID_score, f)

    with open(f"{dirname}/sn_results.pkl", "wb") as f:
        pickle.dump(disc_losses, f)
        pickle.dump(gen_losses, f)

    torch.save(generator.state_dict(), f"{dirname}/sn_generator.pth")
    torch.save(discriminator.state_dict(), f"{dirname}/sn_discriminator.pth")


# In[5]:


# load the saved discriminator
discriminator = DCGAN_Discriminator(IMAGE_CHANNELS, FEATURES_DISC).to(device)
initialize_weights(discriminator)
checkpoint = torch.load("{dirname}/sn_discriminator.pth")
discriminator.load_state_dict(checkpoint[discriminator.state_dict()])


# In[ ]:


# get the discriminator accuracy
accuracy = evaluate_discriminator(discriminator, val_data, val_loader)
print(f"Accuracy: {accuracy}")

