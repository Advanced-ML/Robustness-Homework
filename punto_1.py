import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from models import *
from Adversarial_attacks import pgd
from skimage import io
import matplotlib.pyplot as plt
from tqdm import tqdm 
from skimage import io


#-------------------------------------------Hyperparameters-------------------------------------
"""The parameters shown are those that can be considered as small.  You can also use more steps or a diferent norm,
 but the idea es just to varied the step size and the epsilon.
"""
number_of_steps = 200
step_size = 0.0001
norm = float("inf") # Use Infinity norm as float("inf") or Euclidean norm as 2
epsilon = 0.037

#------------------------------------------------------------------------------------------------

class Initial_Dataset(Dataset):
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


model = resnet50(weights=ResNet50_Weights.DEFAULT)
paths_list = [os.path.join("carro.jpg")]
annots = [479]
batch_size = 1
transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((512, 512)), transforms.ToTensor()])
dataset_images = Initial_Dataset(annots, paths_list, transform = transform)
kwargs = {}
loader = torch.utils.data.DataLoader(dataset_images, batch_size=batch_size, shuffle=False, **kwargs)
device = torch.device('cuda')
model.to(device)
model.eval()
for image, annot in loader:
    adversarial_example = pgd(model, image.cuda(), annot.cuda(), F.cross_entropy, num_steps = number_of_steps, step_size = step_size, grad_norm = norm, eps = epsilon, ball_norm =norm)
    pred = model(adversarial_example.to(device))
    pred = F.softmax(pred, dim = -1).data.max(1)[1]


"""Here you have an example of how you can visualize an adversarial example. Now you have to create the subplot with 9 different experiments.
A way to organize this subplot is proposed in the git hub. However it is not mandatory to do it this way,
just be clear with the parameters used to produce each image."""

new_x = adversarial_example.cpu().numpy()
new_x = new_x[0]
new_x = np.transpose(new_x, axes = (1,2,0))

plt.figure()
plt.imshow(new_x)
plt.axis("off")
plt.show()




