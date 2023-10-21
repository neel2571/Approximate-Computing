    
import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from models_extension import resnet18
from models_extension import LeNet5
#from resnet_approx1 import resnet18
import random
import torchvision
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import Module
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import sys

class AdvDataset(Dataset):

    def __init__(self, model_name, dataset_name, transform=None):
        df = pd.read_csv(f"../results/posit_{model_name}_{dataset_name}_tensor_result.csv")
        self.model_name = model_name
        self.dataset_name = dataset_name
        # self.attack_name = attack_name
        self.img_nums = df['ImgNum']
        self.y = df[' GT']
        self.transform = transform

    def __getitem__(self, index):
        img = torch.load(f"../tensors/{self.model_name}_{self.dataset_name}/original_be_{self.img_nums[index]}_{self.model_name}_{self.dataset_name}.pt")
        

        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]

model_name = sys.argv[1]
dataset_name = sys.argv[2]
# attack_name = sys.argv[3]
approx_design = sys.argv[3]


if model_name == "resnet18":
    model = resnet18()
elif model_name == "lenet":
    model = LeNet5()
else:
    raise Exception("No model with the specified name found!")


model.to('cpu')
checkpoint = torch.load(f'../models/{model_name}_{dataset_name}_be.pth',map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
# model.load_state_dict(checkpoint)


dataset = AdvDataset(model_name, dataset_name)
data_loader = DataLoader(dataset=dataset, batch_size=1)


outfile = open(f"../results/positout_biterror_{model_name}_{dataset_name}_{approx_design}_tensor.csv", "w")
print("running posit")
outfile.write("Count,AdvP,ApxP\n")

model.eval()
count = 0
with torch.no_grad():
  for data in data_loader:
    inputs, labels = data[0].to("cpu"), data[1].to("cpu")
    #print(f"-----------{inputs[0].shape}------------------------")
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    for i in range(inputs.shape[0]):
        outfile.write(f"{count}, {labels[i]}, {predicted[i]}\n")
        count += 1
        # print(count)
outfile.close()
