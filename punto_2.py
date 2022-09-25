import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import linalg as LA
import torchvision
import torchvision.transforms as transforms
import os
from models import *
from torch.autograd import Variable
from tqdm import tqdm
from Adversarial_attacks import pgd, apgd

#-------------------------------------------Hyperparameters-------------------------------------
number_of_steps = 10
step_size = 0.005
norm = float("inf") # Use Infinity norm as float("inf") or Euclidean norm as 2  
epsilon = 0.037
algorithm = "PGD" #Use PGD or APGD to change the adversarial algortithm

#----------------------------------------------------------------------------------------------


device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])


test_dataset = torchvision.datasets.CIFAR10(root=os.path.join("Pytorch-Adversarial-Training-CIFAR", "data"), train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=250, shuffle=False, num_workers=4)

net = ResNet18()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True
load_aux = torch.load('basic_training_final')
net.load_state_dict(load_aux['net'])
criterion = nn.CrossEntropyLoss()


def test():
    net.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    with torch.enable_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            benign_loss += loss.item()
            _, predicted = outputs.max(1)
            benign_correct += predicted.eq(targets).sum().item()
            if algorithm =="PGD":
                adv = pgd(net,inputs, targets, criterion,num_steps = number_of_steps, step_size = step_size, grad_norm = norm, eps = epsilon, ball_norm = norm )
            else: 
                adv = apgd(net,inputs, targets, criterion,num_steps = number_of_steps, step_size = step_size, grad_norm = norm, eps = epsilon, ball_norm = norm )
            adv_outputs = net(adv)
            loss = criterion(adv_outputs, targets)
            adv_loss += loss.item()
            _, predicted = adv_outputs.max(1)
            adv_correct += predicted.eq(targets).sum().item()
    return benign_correct,adv_correct


accuracy_results = test()
print("Algorithm:", algorithm)
print("Accuracy of natural model: ", str((accuracy_results[0]/ len(test_dataset)) * 100) + "%")
print("Accuracy of the model after adversarial attack: ", str((accuracy_results[1] / len(test_dataset)) * 100) + "%")

