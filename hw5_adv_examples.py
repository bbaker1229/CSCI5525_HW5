############################################
# Name: Bryan John Baker
# Student ID: bake1358
# Email: bake1358@umn.edu
# Collaborators:
# File Name: hw5_adv_examples.py
############################################

from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


def get_prediction_name(prediction):
    classes = pd.read_json('imagenet_class_index.json')
    index = torch.argmax(prediction).item()
    return classes[index][1]


# Import pre-trained classifier
from torchvision.models import resnet50
model = resnet50(pretrained=True)
model.eval()
# Set up preprocessor
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
unprocess = transforms.Compose([
    transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225]),
    transforms.ToPILImage(),
    # transforms.Resize(1553),
])

# Import Elephant example
my_img = Image.open("Elephant2.jpg")
my_tensor = preprocess(my_img)[None, :, :, :]
pred_vector = model(my_tensor)
get_prediction_name(pred_vector)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()

old_name = get_prediction_name(pred_vector)
new_name = old_name
epoch = 1
my_tensor.requires_grad = True
loss_lst = []
new_tensor = my_tensor

while (new_name == old_name):
    print('Epoch #: %d' % (epoch))
    old_name = new_name
    my_tensor = new_tensor.detach()
    my_tensor.requires_grad = True
    pred_vector = model(my_tensor)
    loss = criterion(pred_vector, torch.LongTensor([101]))
    model.zero_grad()
    loss.backward()
    image_gradient = my_tensor.grad.data  # Get gradients of image
    image_gradient = torch.clamp(image_gradient, min=-0.05, max=0.05)
    new_tensor = my_tensor + 0.01 * image_gradient  # usually we go down the gradient, here we go up.
    pred_vector = model(new_tensor)
    new_name = get_prediction_name(pred_vector)
    loss_lst.append(loss.item())
    print("Loss: %f" % (loss.item()))
    epoch += 1
print("The image is now misclassified.")

# What is the new label:
print(new_name)

# Display the new image:
new_image = unprocess(new_tensor[0])
new_image.show()

############################################################
############################################################
############################################################

# Make the image predict "bullet_train"
# Import Elephant example
my_img = Image.open("Elephant2.jpg")
my_tensor = preprocess(my_img)[None, :, :, :]
pred_vector = model(my_tensor)
get_prediction_name(pred_vector)


#Define the loss function
def loss_function(predict):
    to_bullet = F.cross_entropy(predict, torch.LongTensor([466]), reduction='sum')
    to_tusk = F.cross_entropy(predict, torch.LongTensor([101]), reduction='sum')
    return to_bullet - to_tusk


old_name = get_prediction_name(pred_vector)
new_name = old_name
epoch = 1
my_tensor.requires_grad = True
loss_lst = []
new_tensor = my_tensor

while (new_name != 'bullet_train'):
    print('Epoch #: %d' % (epoch))
    old_name = new_name
    my_tensor = new_tensor.detach()
    my_tensor.requires_grad = True
    pred_vector = model(my_tensor)
    loss = loss_function(pred_vector)
    model.zero_grad()
    loss.backward()
    image_gradient = my_tensor.grad.data  # Get gradients of image
    image_gradient = torch.clamp(image_gradient, min=-0.05, max=0.05)
    new_tensor = my_tensor - 0.01 * image_gradient
    pred_vector = model(new_tensor)
    new_name = get_prediction_name(pred_vector)
    loss_lst.append(loss.item())
    print("Loss: %f" % (loss.item()))
    epoch += 1
print("The image is now a bullet train.")

# What is the new label:
print(new_name)

# Display the new image:
new_image = unprocess(new_tensor[0])
new_image.show()
