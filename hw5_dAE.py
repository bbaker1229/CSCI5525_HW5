############################################
# Name: Bryan John Baker
# Student ID: bake1358
# Email: bake1358@umn.edu
# Collaborators:
# File Name: hw5_dAE.py
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


# Define the dAE Network
class dAE(nn.Module):
    def __init__(self):
        super(dAE, self).__init__()
        self.fc1 = nn.Linear(784, 400, bias=False)
        self.fc2 = nn.Linear(400, 20, bias=False)
        self.fc3 = nn.Linear(20, 400, bias=False)
        self.fc4 = nn.Linear(400, 784, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.sig(self.fc4(x))
        return x


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
    train_load = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True, num_workers=0)
    test_load = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False, num_workers=0)
    return train_load, test_load


# Get data set and prepare data loaders for torch
train_set, test_set = get_mnist_data()
train_loader, test_loader = load_mnist_data(train_set, test_set)

# Set up Neural Network
denoiseAE = dAE()

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(denoiseAE.parameters(), lr=0.001)

# Record the batch loss and the training accuracy for each epoch
denoise_loss_lst = []
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
        z = torch.randn(labels.shape[0], 784, dtype=torch.float).reshape(-1, 1, 28, 28)
        # noisy = (images + z) / 2.0
        noisy = images + z
        output = denoiseAE(noisy)
        loss = criterion(output.reshape(-1, 1, 28, 28), images)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        image_total += labels.size(0)
        batch_total += 1.0
    denoise_loss_lst.append(running_loss / batch_total)
    print('Loss: %.4f' % (denoise_loss_lst[epoch]))
print("Finished Training")

# Plot Average Loss
plt.plot(denoise_loss_lst)
plt.title("Denoising AE Average Loss by Epoch")
plt.ylabel("Average Loss")
plt.xlabel("Epoch")
plt.show()

# Save the trained neural networks
PATH = './hw5_dAE.pth'
torch.save(denoiseAE.state_dict(), PATH)

for i, data in enumerate(test_loader, 0):
    images, labels = data
    z = torch.randn(labels.shape[0], 784, dtype=torch.float).reshape(-1, 1, 28, 28)
    noisy = images + z

initial_data = noisy[0:5]
final_data = denoiseAE(noisy).reshape(-1, 1, 28, 28)[0:5]
final_data = torch.cat((initial_data, final_data))
imshow(torchvision.utils.make_grid(final_data, nrow=5))

