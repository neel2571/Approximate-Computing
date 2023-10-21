import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset
import torchvision
import torchvision.models as models
from torchvision.utils import save_image
import torchvision.transforms as transforms
import numpy as np
from torch.nn import Module

# if __name__ == "main":

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device.")

batch_size = 256

model = models.vgg16(pretrained=True)
# cpt = torch.load("../models/vgg16-397923af.pth", map_location = torch.device('cpu'))
# model.load_state_dict(cpt['model_state_dict'])
model.eval()
model.to(device)

transform = transforms.Compose([transforms.ToTensor()])
data_set = torchvision.datasets.ImageNet(root='../datasets/IMAGENET', train=False, download=True, transform = transform)
data_loader = torch.utils.data.DataLoader(data_set, batch_size = batch_size)

results = open.write("ImgNum, GT, NP\n")

count = 0
for data in data_loader:
    inputs, labels = data[0].to(device), data[1].to(device)
    outputs = model(inputs)
    print(outputs.data.shape)
    _, predicted = torch.max(outputs.data, 1)
    for i in range(inputs.shape[0]):
        torch.save(inputs[i], f"../tensors/vgg16_imagenet/original_{count}_vgg16_imagenet.pt")
        results.write(f"{count}, {labels[i]}, {predicted[i]}\n")
        count +=1
    print(count)
    if count == 608:
        break

results.close()