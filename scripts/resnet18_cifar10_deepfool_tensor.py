import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
import torchvision.models as models
from torchvision.utils import save_image
import torchvision.transforms as transforms
import numpy as np
from torch.nn import Module

def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")


    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, label, k_i, pert_image

if __name__ == "__main__":

    #Uses GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device.')

    batch_size = 1

    model = models.resnet18()
    cpt = torch.load("../models/resnet18_cifar10", map_location=torch.device('cpu'))
    model.load_state_dict(cpt['model_state_dict'])
    model.eval()
    model.to(device)

    transform = transforms.Compose([transforms.ToTensor()])
    data_set = torchvision.datasets.CIFAR10(root='../datasets/', train=False, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size)

    results = open("../results/deepfool_resnet18_cifar10_tensor_result.csv", "w")
    results.write("ImgNum,GT,NP,AdvP,Iters\n")

    count = 0
    for data in data_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        r_tot, loop_i, label, k_i, pert_image = deepfool(inputs[0], model)
        # adv_outputs = model(adv_imgs)
        # _, adv_predicted = torch.max(adv_outputs.data, 1)
        for i in range(inputs.shape[0]):
            print(pert_image.shape)
            torch.save(pert_image, f"../tensors/resnet18_cifar10/deepfool_{count}_resnet18_cifar10.pt")
            results.write(f"{count}, {labels[i]}, {predicted[i]}, {k_i}, {loop_i}\n")
            count += 1
        print(count)
    results.close()


