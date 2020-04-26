############################################
# Name: Bryan John Baker
# Student ID: bake1358
# Email: bake1358@umn.edu
# Collaborators:
# File Name: hw5_gan.py
############################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# Define the Generator Network
class GenNet(nn.Module):
    def __init__(self):
        super(GenNet, self).__init__()
        self.fc1 = nn.Linear(128, 256, bias=False)
        self.fc2 = nn.Linear(256, 512, bias=False)
        self.fc3 = nn.Linear(512, 784, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.view(-1, 128)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.tanh(self.fc3(x))
        return x


# Define the Discriminator Network
class DiscNet(nn.Module):
    def __init__(self):
        super(DiscNet, self).__init__()
        self.fc1 = nn.Linear(784, 512, bias=False)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.fc3 = nn.Linear(256, 1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.sig(self.fc3(x))
        return x


def get_mnist_data():
    """
    Get the MNIST dataset from torchvision.
    :return: Train data set.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, ), (0.5, ))])
    train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return train


def load_mnist_data(train):
    """
    Create training and test dataset loaders.
    :param train: Training dataset.
    :return: training data loader.
    """
    train_load = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True, num_workers=0)
    return train_load


def imshow(img):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Get data set and prepare data loaders for torch
train_set = get_mnist_data()
train_loader = load_mnist_data(train_set)

# Set up Neural Networks
gennet = GenNet()
discnet = DiscNet()

# Define the loss function and optimizer
criterion = nn.BCELoss()
genoptimizer = optim.Adam(gennet.parameters(), lr=0.001)
discoptimizer = optim.Adam(discnet.parameters(), lr=0.001)

# Record the batch loss and the training accuracy for each epoch
disc_loss_lst = []
gen_loss_lst = []
for epoch in range(50):
    print('Epoch #: %d' % (epoch + 1))
    running_discloss = 0.0
    running_genloss = 0.0
    batch_total = 0.0
    image_total = 0.0
    correct = 0.0
    for i, data in enumerate(train_loader, 0):
        # Train Discriminator:
        # get the inputs; data is a list of [inputs, labels]
        real_images, labels = data
        # zero the parameter gradients
        discoptimizer.zero_grad()
        # forward + backward + optimize
        z = torch.randn(100, 128, dtype=torch.float)
        fake_images = gennet(z)
        real_outputs = discnet(real_images)
        fake_outputs = discnet(fake_images.detach())
        outputs = torch.cat((real_outputs, fake_outputs), 0)
        real = torch.as_tensor(np.array([1 for i in range(100)]))
        fake = torch.as_tensor(np.array([0 for i in range(100)]))
        labels = torch.cat((real, fake), 0)
        discloss = criterion(outputs, labels * 1.0)
        disclossreal = criterion(real_outputs, real * 1.0)
        disclossreal.backward()
        disclossfake = criterion(fake_outputs, fake * 1.0)
        disclossfake.backward()
        discloss = disclossreal + disclossfake
        discoptimizer.step()
        # discloss.backward(retain_graph=True)
        # discloss.backward()

        # Train Generator:
        # zero the parameter gradients
        genoptimizer.zero_grad()
        fake_outputs = discnet(fake_images)
        genloss = criterion(fake_outputs, real * 1.0)
        genloss.backward()
        genoptimizer.step()

        # print statistics
        running_discloss += discloss.item()
        running_genloss += genloss.item()
        image_total += labels.size(0)
        batch_total += 1.0
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    disc_loss_lst.append(running_discloss / batch_total)
    gen_loss_lst.append(running_genloss / batch_total)
    print('Gen Loss: %.4f, Disc Loss: %.4f' % (gen_loss_lst[epoch], disc_loss_lst[epoch]))
    if ((epoch + 1) % 10) == 0:
        imshow(torchvision.utils.make_grid(fake_images[0:16].reshape(-1, 1, 28, 28), nrow=4))
print("Finished Training")

# Plot Discriminator Loss
plt.plot(disc_loss_lst)
plt.title("GAN Discriminator Loss by Epoch")
plt.ylabel("Binary Cross Entropy Loss")
plt.xlabel("Epoch")
plt.show()

# Plot Generator Loss
plt.plot(gen_loss_lst)
plt.title("GAN Generator Loss by Epoch")
plt.ylabel("Binary Cross Entropy Loss")
plt.xlabel("Epoch")
plt.show()

# Save the trained neural networks
PATH = './hw5_gan_dis.pth'
torch.save(discnet.state_dict(), PATH)
PATH = './hw5_gan_gen.pth'
torch.save(gennet.state_dict(), PATH)

# Image Samples
imshow(torchvision.utils.make_grid(real_images[0:16], nrow=4))
imshow(torchvision.utils.make_grid(fake_images[0:4].reshape(-1, 1, 28, 28), nrow=4))
