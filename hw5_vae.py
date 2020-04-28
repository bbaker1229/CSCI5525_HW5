############################################
# Name: Bryan John Baker
# Student ID: bake1358
# Email: bake1358@umn.edu
# Collaborators:
# File Name: hw5_vae.py
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


# Define the VAE Network
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fcmu = nn.Linear(400, 20)
        self.fcvar = nn.Linear(400, 20)
        self.fc2 = nn.Linear(20, 400)
        self.fc3 = nn.Linear(400, 784)
        self.sig = nn.Sigmoid()

    def forward(self, x, new, new_z):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        xmu = self.fcmu(x)
        xvar = self.fcvar(x)
        xvar = torch.abs(xvar)
        m = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))
        xsig = torch.sqrt(xvar)
        xsig = torch.mul(xsig, m.sample(xsig.shape))
        xsig = torch.abs(xsig)
        z = xsig + xmu
        # z = torch.distributions.Normal(xmu, xsig)
        # z = z.sample()
        if new:
            z = new_z
        x = F.relu(self.fc2(z))
        x = self.sig(self.fc3(x))
        return x, xmu, xvar


def imshow(img):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_mnist_data():
    """
    Get the MNIST dataset from torchvision.
    :return: Train and test data sets.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, ), (0.5, ))])
    train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train, test


def load_mnist_data(train, test):
    """
    Create training and test dataset loaders.
    :param train: Training dataset.
    :param test: Test dataset.
    :return: training and test data loaders.
    """
    train_load = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True, num_workers=0)
    test_load = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False, num_workers=0)
    return train_load, test_load


def loss_function(predict, actual, mu, var):
    bceloss = F.binary_cross_entropy(predict, actual.view(-1, 784), reduction='sum')
    KL = 0.5 * torch.sum(1 + torch.log(var) - torch.pow(mu, 2) - var)
    # KL = KL / predict.shape[0] * 784
    return bceloss - KL

# Get data set and prepare data loaders for torch
train_set, test_set = get_mnist_data()
train_loader, test_loader = load_mnist_data(train_set, test_set)

# Set up Neural Network
vaenet = VAE()

# Define the optimizer
optimizer = optim.Adam(vaenet.parameters(), lr=0.001)

# Record the batch loss and the training accuracy for each epoch
vae_loss_lst = []
for epoch in range(10):
    print('Epoch #: %d' % (epoch + 1))
    running_loss = 0.0
    batch_total = 0.0
    image_total = 0.0
    correct = 0.0
    for i, data in enumerate(train_loader, 0):
        # Train Encoder:
        # get the inputs; data is a list of [inputs, labels]
        images, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output, mu, var = vaenet(images, False, 1)
        loss = loss_function(output, images, mu, var)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        image_total += labels.size(0)
        batch_total += 1.0
    vae_loss_lst.append(running_loss / batch_total)
    print('Loss: %.4f' % (vae_loss_lst[epoch]))
print("Finished Training")

# Plot Average Loss
plt.plot(vae_loss_lst)
plt.title("VAE Average Loss by Epoch")
plt.ylabel("Average Loss")
plt.xlabel("Epoch")
plt.show()

# Save the trained neural network
PATH = './hw5_vae.pth'
torch.save(vaenet.state_dict(), PATH)

# Create a test set
test_images = images[0:16]
z = torch.randn(16, 20)
output_images, _, _ = vaenet(test_images, True, z)
output_images = output_images.reshape(-1, 1, 28, 28)

imshow(torchvision.utils.make_grid(output_images, nrow=4))
