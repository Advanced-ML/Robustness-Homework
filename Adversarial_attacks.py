import numpy as np
import torch
from torch import linalg as LA

#-----------------------------------------------------PGD------------------------------------------------------------
def pgd(model, x, y, loss_fn, num_steps, step_size, grad_norm, eps, ball_norm, clamp=(0,1), y_target=None):
    x_adversarial = x.clone().detach().requires_grad_(False).to(x.device)
    targeted = y_target is not None
    step_counter = 0
    if targeted:
        y_target = (np.ones(y.shape[0])*y_target).astype(int)
        y_target = torch.tensor(y_target).to(x.device)
    while True:
        with torch.enable_grad():
            x_adv_aux = x_adversarial.clone().detach().requires_grad_(True)
            prediction = model(x_adv_aux)
            loss = loss_fn(prediction, y_target if targeted else y)
            loss.backward()
            gradient_num = x_adv_aux.grad * step_size
            with torch.no_grad():
                gradient_den = LA.norm(x_adv_aux.grad.view(x_adv_aux.shape[0], -1), grad_norm, dim=-1) + 1e-12
                gradient = gradient_num * ((1/gradient_den).view(-1,1,1,1))
                if targeted:
                    x_adversarial -= gradient
                else:
                    x_adversarial += gradient
            if ball_norm == float('inf'):
                x_adversarial = torch.max(torch.min(x_adversarial, x + eps), x - eps)
            else: 
                delta = x_adversarial - x
                norms = delta.view(delta.shape[0], -1).norm(ball_norm, dim=1) 
                outside_deltas = norms <= eps
                norms[outside_deltas] = eps
                delta = (delta/norms.view(-1,1,1,1)) * eps
                x_adversarial = x + delta
            step_counter += 1
            x_adversarial = x_adversarial.clamp(*clamp)
            if step_counter == num_steps:
                break
    return x_adversarial.detach()


#--------------------------------------APGD----------------------------------------------------

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
    return x_adversarial

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
    
def make_step(x_adversarial, model, loss_fn, y_target, targeted, y, step_size, grad_norm):
    x_adv_aux = x_adversarial.clone().detach().requires_grad_(True)
    x_adversarial_aux = x_adversarial.clone().detach()
    prediction = model(x_adv_aux)
    if targeted:
        loss_fn = dlr_loss_targeted
        loss = loss_fn(prediction, y, y_target)
    else:
        loss = loss_fn(prediction, y)
    loss_aux = torch.sum(loss) 
    loss_aux.backward()
    gradient = gradiente(x_adv_aux, step_size, grad_norm)
    x_adversarial_aux += gradient
    return x_adversarial_aux.clone().detach(), loss.detach().requires_grad_(False)

def dlr_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    u = torch.arange(x.shape[0])

    return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
        1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

def dlr_loss_targeted(x, y,y_target):
        x_sorted, ind_sorted = x.sort(dim=1)
        u = torch.arange(x.shape[0])

        return -(x[u, y] - x[u, y_target]) / (x_sorted[:, -1] - .5 * (
            x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)

def apgd(model, x, y, loss_fn, num_steps, step_size, grad_norm, eps, ball_norm, alpha=0.75, clamp=(0,1), y_target=None):
    exitosos = torch.zeros(x.shape[0]).to(x.device)
    step_size = torch.ones(x.shape[0]).to(x.device) * step_size
    step_size_0 = step_size.clone().detach()
    step_size_1 = step_size.clone().detach()
    W = checkpoints_list_generator(num_steps)
    x_adversarial_0 = x.clone().detach().requires_grad_(False).to(x.device)
    targeted = y_target is not None

    x_adversarial_1, loss_0 = make_step(x_adversarial_0, model, loss_fn, y_target, targeted, y, step_size, grad_norm)
    loss_0 = loss_0.data
    x_adversarial_1 = projection(x, x_adversarial_1, eps, ball_norm)
    x_adversarial_1 = x_adversarial_1.clamp(*clamp)
    x_adversarial_2 = None
    x_max = None
    x_max_0 = None
    step_counter = 1
    if targeted:
        loss_fn = dlr_loss_targeted
        loss_1 = loss_fn(model(x_adversarial_1), y, y_target)
    else:
        loss_1 = loss_fn(model(x_adversarial_1), y)
    loss_1 = loss_1.data
    loss1_ = loss_1.clone().detach()
    loss_max = loss_0.clone().requires_grad_(False)
    loss_max[loss_0<loss_1] = loss1_[loss_0<loss_1]
    loss_max_0 = loss_max.clone().requires_grad_(False)
    loss_max_1 = loss_max.clone().requires_grad_(False)
    x_max = x_adversarial_0 * ((loss_0>loss_1).view(-1,1,1,1)) + x_adversarial_1 * ((loss_0<loss_1).view(-1,1,1,1))

    while True:
        z, loss_1 = make_step(x_adversarial_1, model, loss_fn, y_target, targeted, y, step_size, grad_norm)
        z = projection(x, z, eps, ball_norm)
        x_adv_2_aux = x_adversarial_1 + (alpha*(z - x_adversarial_1)) + ((1-alpha)*(x_adversarial_1 -  x_adversarial_0))
        x_adversarial_2 = projection(x, x_adv_2_aux, eps, ball_norm)
        x_adversarial_2 = x_adversarial_2.clamp(*clamp)
        exitosos += exitoso(loss_0, loss_1)
        if x_max_0 == None:
            loss_max = loss_1.clone().detach()
            x_max = x_adversarial_1.clone().requires_grad_(False)
            x_max_0 = x_adversarial_0.clone().requires_grad_(False)
        else:
            x_max = x_max * ((loss_max>=loss_1).view(-1,1,1,1)) + x_adversarial_1 * ((loss_max<loss_1).view(-1,1,1,1))
            x_max_0 = x_max_0 * ((loss_max>=loss_1).view(-1,1,1,1)) + x_adversarial_0 * ((loss_max<loss_1).view(-1,1,1,1))
        loss1_ = loss_1.clone().detach()
        loss_max[loss_1>loss_max] = loss1_[loss_1>loss_max]
        if step_counter in W:
            total_steps = W[1] - W[0] 
            W = W[1:]
            loss_max_1 = loss_max.clone().detach()
            a = A(exitosos, total_steps)
            b = B(step_size_0, step_size_1, loss_max_0, loss_max_1)
            step_size = step_size * ((a == False) * (b == False)) + step_size/2 * ((a + b)>0)
            x_adversarial_2 = x_adversarial_2 * (((a == False) * (b == False)).view(-1,1,1,1)) + x_max * (((a + b)>0).view(-1,1,1,1))
            x_adversarial_1 = x_adversarial_1 * (((a == False) * (b == False)).view(-1,1,1,1)) + x_max_0 * (((a + b)>0).view(-1,1,1,1))
            step_size_0 = step_size_1.clone().detach()
            step_size_1 = step_size.clone().detach()
            loss_max_0 = loss_max_1.clone().detach()
            exitosos = torch.zeros(x.shape[0]).to(x.device)
        
        x_adversarial_0 = x_adversarial_1.clone().detach()
        x_adversarial_1 = x_adversarial_2.clone().detach()
        loss_0 = loss_1.clone().detach()
        step_counter += 1
        if step_counter == num_steps:
            break
    return x_max.detach()
