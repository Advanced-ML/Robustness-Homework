import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from skimage import io
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
from torch import linalg as LA


class images(Dataset):
    def __init__(self, annotations, images_paths: list, transform=None):
        self.annotations = annotations
        self.images_paths = images_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = self.images_paths[index]
        x = io.imread(img_path)
        x = self.transform(x)
        y = torch.tensor(self.annotations[index])
        return x, y



def projection(x, x_adversarial, eps, ball_norm):
    if ball_norm == float('inf'):
        x_adversarial = torch.max(torch.min(x_adversarial, x + eps), x - eps)
    else: 
        delta = x_adversarial - x
        norms = delta.view(delta.shape[0], -1).norm(ball_norm, dim=1)
        outside_deltas = norms <= eps
        norms[outside_deltas] = eps
        delta = (delta/norms.view(-1,1,1,1) + 1e-12) * eps
        x_adversarial = x + delta
    return x_adversarial.requires_grad_(False)


def ps_generator():
    ps_list = []
    p0 = 0 
    p1 = 0.22
    pj_1 = p0
    pj = p1
    ps_list.append(p0)
    ps_list.append(p1)
    while True:
        pj1 = pj + max(pj - pj_1 - 0.03, 0.06)
        if pj1 >1:
            break
        ps_list.append(pj1)
        pj_1 = pj
        pj = pj1
    return ps_list
 
def checkpoints_list_generator(num_steps):
    lista_ps = np.array(ps_generator())
    lista_checkpoints= np.unique(np.ceil(lista_ps * num_steps)).astype(int)
    return lista_checkpoints

def A(exitosos, total_steps, ro = 0.75):
    desired = (total_steps) * ro
    return exitosos < desired

def exitoso(loss_0, loss_1):
    return loss_1 > loss_0

def B(step_size_0, step_size_1, loss_max_0, loss_max_1):
    bool_1 = step_size_0 == step_size_1
    bool_2 = loss_max_0 == loss_max_1
    return bool_1 * bool_2

def gradiente(x_adv_aux, step_size, grad_norm):
    with torch.no_grad():
        gradient_num = x_adv_aux.grad * step_size.view(-1,1,1,1)
        gradient_den = LA.norm(x_adv_aux.grad.view(x_adv_aux.shape[0], -1), grad_norm, dim=-1) + 1e-12
        gradient = gradient_num * ((1/gradient_den).view(-1,1,1,1))
    return gradient


def make_step(x_adversarial, model, loss_fn, c, y, step_size, grad_norm):
    x_adv_aux = x_adversarial.clone().detach().requires_grad_(True)
    x_adversarial_aux = x_adversarial.clone().detach().requires_grad_(False)
    prediction = model(x_adv_aux)
    loss = loss_fn(prediction, y, c)
    loss_aux = torch.sum(loss) 
    loss_aux.backward()
    gradient = gradiente(x_adv_aux, step_size, grad_norm)
    x_adversarial_aux += gradient.clone().detach().requires_grad_(False)
    return x_adversarial_aux.clone().detach().requires_grad_(False), loss



def K_calculator(x, y, ks):
    x_sorted, ind_sorted = x.sort(dim=1)
    c = ind_sorted[ind_sorted != y.view(ind_sorted.shape[0],-1)].view(ind_sorted.shape[0],-1).requires_grad_(False)
    k = c[:,-ks:]
    return k.requires_grad_(False)

def mm_loss(x,y,c):
    u = torch.arange(x.shape[0]).requires_grad_(False)
    return -x[u, y] +  x[u, c]

def mm_attack(model, x, y, loss_fn, num_steps, step_size, grad_norm, eps, ball_norm, ks, clamp=(0,1)):
    pred = model(x).detach()
    K = K_calculator(pred,y,ks).requires_grad_(False)
    adversarial_images = None
    adversarial_annots = None 
    while ks > 0:
        W = checkpoints_list_generator(num_steps)
        step_size_ = torch.ones(y.shape[0]).requires_grad_(False).to(x.device) * step_size
        exitosos = torch.zeros(y.shape[0]).requires_grad_(False).to(x.device)
        x_adversarial_0 = x.clone().detach().requires_grad_(False).to(x.device)
        step_size_0 = step_size_.clone().detach().requires_grad_(False)
        step_size_1 = step_size_.clone().detach().requires_grad_(False)
        x_max = x_adversarial_0.clone().detach().requires_grad_(False)
        c = K[:,-1]
        loss_0 = loss_fn(pred,y,c).detach().requires_grad_(False)
        loss__1 = loss_0.clone().detach().requires_grad_(False)
        loss_max =loss_0.clone().detach().requires_grad_(False)
        loss_max_0 = loss_max.clone().detach().requires_grad_(False)
        loss_max_1 = loss_max.clone().detach().requires_grad_(False)
        step_counter = 1
        while True: 
            x_adversarial_1, loss_0 = make_step(x_adversarial_0, model, loss_fn, c, y, step_size_, grad_norm)
            loss_0 = loss_0.clone().detach().requires_grad_(False)
            exitosos += exitoso(loss__1, loss_0)
            x_adversarial_1 = projection(x, x_adversarial_1, eps, ball_norm)
            x_max = x_max * ((loss_max>=loss_0).view(-1,1,1,1)) + x_adversarial_0 * ((loss_max<loss_0).view(-1,1,1,1))
            loss_max[loss_0>loss_max] = loss_0[loss_0>loss_max]
            if step_counter in W:
                total_steps = W[1] - W[0] 
                W = W[1:]
                loss_max_1 = loss_max.clone().detach().requires_grad_(False)
                #---------------------------------------------------------------------------------------------------------
                # In your report explain the lines below, include information about a and b variables and specify what are they doing.
                a = A(exitosos, total_steps)
                b = B(step_size_0, step_size_1, loss_max_0, loss_max_1).detach()
                step_size_ = step_size_ * ((a == False) * (b == False)) + step_size_/2 * ((a + b)>0)
                x_adversarial_1 = x_adversarial_1 * (((a == False) * (b == False)).view(-1,1,1,1)) + x_max * (((a + b)>0).view(-1,1,1,1)) 
                #--------------------------------------------------------------------------------------------------------
                step_size_0 = step_size_1.clone().detach().requires_grad_(False)
                step_size_1 = step_size_.clone().detach().requires_grad_(False)
                loss_max_0 = loss_max_1.clone().detach().requires_grad_(False)
                exitosos = torch.zeros(x.shape[0]).requires_grad_(False).to(x.device)
            x_adversarial_0 = x_adversarial_1.clone().detach().requires_grad_(False)
            loss__1 = loss_0.clone().detach().requires_grad_(False)
            step_counter += 1
            if step_counter == num_steps:
                break

        #TODO: add a line that removes the class c of the set of possible target classes K. Remember that this function is designed to work with more than one image.
        # Hint: see the pseudocode of MMA and review the auxiliar function K_calculator


        predictions = model(x_max).detach()
        predictions = F.softmax(predictions, dim=1).data.max(1)[1]
        if adversarial_images == None:
            adversarial_images = x_max[(predictions != y).view(-1,1,1,1).expand(-1,x.shape[1],x.shape[2],x.shape[3])].view(-1,x.shape[1],x.shape[2],x.shape[3]).requires_grad_(False)
            adversarial_annots = y[(predictions != y)].requires_grad_(False)
        else:
            adversarial_images_aux = x_max[(predictions != y).view(-1,1,1,1).expand(-1,x.shape[1],x.shape[2],x.shape[3])].view(-1,x.shape[1],x.shape[2],x.shape[3]).requires_grad_(False)
            adversarial_images = torch.cat((adversarial_images,adversarial_images_aux),0).requires_grad_(False)
            adversarial_annots = torch.cat((adversarial_annots,y[(predictions != y)]),0).requires_grad_(False)
        x = x[(predictions == y).view(-1,1,1,1).expand(-1,x.shape[1],x.shape[2],x.shape[3])].view(-1,x.shape[1],x.shape[2],x.shape[3]).requires_grad_(False) 
        ks -= 1
        pred = model(x).detach()
        exitosos = torch.zeros(x.shape[0]).requires_grad_(False).to(x.device)
        y = y[predictions == y].requires_grad_(False)
        if pred.shape[0] == 0 or K.shape[1] == 0:
            if x.shape[0] != 0:
                adversarial_images_aux = x_max[(predictions == y).view(-1,1,1,1).expand(-1,x.shape[1],x.shape[2],x.shape[3])].view(-1,x.shape[1],x.shape[2],x.shape[3]).requires_grad_(False)
                adversarial_images = torch.cat((adversarial_images,adversarial_images_aux),0).requires_grad_(False)
                adversarial_annots = torch.cat((adversarial_annots,y[(predictions == y)]),0).requires_grad_(False)
            break
        K = K[(predictions == y).view(-1,1).expand(-1,K.shape[1])].view(-1,K.shape[1]).requires_grad_(False)
    return adversarial_images, adversarial_annots



transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((512, 512)), transforms.ToTensor()])
dataset_images = images([479], [ os.path.join("carro.jpg")], transform = transform)

kwargs = {}
loader = torch.utils.data.DataLoader(dataset_images, batch_size=1, shuffle=False, **kwargs)

device = torch.device('cuda')
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.to(device)
model.eval()

for image, annot in loader:
    new_x, annotations = mm_attack(model, image.cuda().requires_grad_(False), annot.cuda().requires_grad_(False), mm_loss, num_steps = 40, step_size = 0.001, grad_norm = float("inf"), eps = 0.05, ball_norm = float("inf"),ks=3)
    print("Initial prediction:", F.softmax(model(image.cuda()), dim=1).data.max(1)[1].cpu().numpy()[0])
    print("Final prediction:", F.softmax(model(new_x), dim=1).data.max(1)[1].cpu().numpy()[0])