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


# Define the Encoder Network
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784, 400, bias=False)
        self.fc2 = nn.Linear(400, 20, bias=False)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


# Define the Decoder Network
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(20, 400, bias=False)
        self.fc2 = nn.Linear(400, 784, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 20)
        x = F.relu(self.fc1(x))
        x = self.sig(self.fc2(x))
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

# Set up Neural Networks
encode = Encoder()
decode = Decoder()

# Define the loss function and optimizer
Ecriterion = nn.CrossEntropyLoss()
Dcriterion = nn.BCELoss()
Eoptimizer = optim.Adam(encode.parameters(), lr=0.001)
Doptimizer = optim.Adam(decode.parameters(), lr=0.001)

# Record the batch loss and the training accuracy for each epoch
decode_loss_lst = []
encode_loss_lst = []
for epoch in range(10):
    print('Epoch #: %d' % (epoch + 1))
    running_encode_loss = 0.0
    running_decode_loss = 0.0
    batch_total = 0.0
    image_total = 0.0
    correct = 0.0
    for i, data in enumerate(train_loader, 0):
        # Train Encoder:
        # get the inputs; data is a list of [inputs, labels]
        images, labels = data
        # zero the parameter gradients
        Eoptimizer.zero_grad()
        # forward + backward + optimize
        z = torch.randn(labels.shape[0], 784, dtype=torch.float).reshape(-1, 1, 28, 28)
        # noisy = (images + z) / 2.0
        noisy = images + z
        encoded_output = encode(noisy)
        encodeloss = Ecriterion(encoded_output, labels)
        encodeloss.backward()
        Eoptimizer.step()

        # Train Decoder:
        Doptimizer.zero_grad()
        encoded_output = encode(noisy)
        decoded_output = decode(encoded_output)
        decodeloss = Dcriterion(decoded_output.reshape(-1, 1, 28, 28), images)
        decodeloss.backward()
        Doptimizer.step()

        # print statistics
        running_decode_loss += decodeloss.item()
        running_encode_loss += encodeloss.item()
        image_total += labels.size(0)
        batch_total += 1.0
    decode_loss_lst.append(running_decode_loss / batch_total)
    encode_loss_lst.append(running_encode_loss / batch_total)
    print('Encode Loss: %.4f, Decode Loss: %.4f, Average Loss: %.4f' % (encode_loss_lst[epoch], decode_loss_lst[epoch], (encode_loss_lst[epoch] + decode_loss_lst[epoch]) / 2))
print("Finished Training")

# Plot Average Loss
plt.plot(np.divide(np.array(encode_loss_lst) + np.array(decode_loss_lst), 2.0))
plt.title("Denoising AE Average Loss by Epoch")
plt.ylabel("Average Loss")
plt.xlabel("Epoch")
plt.show()

imshow(torchvision.utils.make_grid(noisy[0:16], nrow=4))
imshow(torchvision.utils.make_grid(decode(encode(noisy)).reshape(-1, 1, 28, 28)[0:16], nrow=4))

# Save the trained neural networks
PATH = './hw5_dAE.pth'
torch.save(decode.state_dict(), PATH)
PATH = './hw5_eAE.pth'
torch.save(encode.state_dict(), PATH)

for i, data in enumerate(test_loader, 0):
    images, labels = data
    z = torch.randn(labels.shape[0], 784, dtype=torch.float).reshape(-1, 1, 28, 28)
    # noisy = (images + z) / 2.0
    noisy = images + z

imshow(torchvision.utils.make_grid(noisy[0:5], nrow=1))
imshow(torchvision.utils.make_grid(decode(encode(noisy)).reshape(-1, 1, 28, 28)[0:5], nrow=1))
