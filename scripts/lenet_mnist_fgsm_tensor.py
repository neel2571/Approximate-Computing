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
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method

class LeNet(Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        # print(y.shape)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y


if __name__ == "__main__":

    eps = 0.25

    #Uses GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device.')

    batch_size = 256

    model = LeNet()
    cpt = torch.load("../models/lenet_mnist", map_location=torch.device('cpu'))
    model.load_state_dict(cpt['model_state_dict'])
    model.eval()
    model.to(device)

    transform = transforms.Compose([transforms.ToTensor()])
    data_set = torchvision.datasets.MNIST(root='../datasets/MNIST/', train=False, transform=transform)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size)

    results = open("../results/fgsm_lenet_mnist_tensor_result.csv", "w")
    results.write("ImgNum,GT,NP,AdvP\n")

    count = 0
    for data in data_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        adv_imgs = fast_gradient_method(model, inputs, eps, np.inf)
        adv_outputs = model(adv_imgs)
        _, adv_predicted = torch.max(adv_outputs.data, 1)
        for i in range(inputs.shape[0]):
            torch.save(adv_imgs[i], f"../tensors/lenet_mnist/fgsm_{count}_lenet_mnist.pt")
            #save_image(inputs[i], f"../imgs/lenet_mnist/test_{count}_lenet_mnist.png")
            results.write(f"{count}, {labels[i]}, {predicted[i]}, {adv_predicted[i]}\n")
            count += 1
        print(count)
    results.close()

