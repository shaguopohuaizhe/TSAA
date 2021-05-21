import argparse
import os
import numpy as np

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from generators import GeneratorResnet
import time
import random 
from shutil import copyfile

seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

parser = argparse.ArgumentParser(description='Training code for generating sparse adversarial examples')
parser.add_argument('--train_dir', default='imagenet', help='path to imagenet training set')
parser.add_argument('--model_type', type=str, default='incv3',
                    help='Model against GAN is trained: incv3, res50')
parser.add_argument('--eps', type=int, default=255, help='Perturbation Budget')
parser.add_argument('--target', type=int, default=-1, help='-1 if untargeted')
parser.add_argument('--batch_size', type=int, default=20, help='Number of trainig samples/batch')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for adam')

args = parser.parse_args()
print(args)

# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Model
if args.model_type == 'incv3':
    model = torchvision.models.inception_v3(pretrained=True)
elif args.model_type == 'res50':
    model = torchvision.models.resnet50(pretrained=True)
model = model.to(device)
model.eval()

# Input dimensions
if args.model_type in ['res50']:
    scale_size = 256
    img_size = 224
else:
    scale_size = 300
    img_size = 299

# Generator
if args.model_type == 'incv3':
    netG = GeneratorResnet(inception=True, eps=args.eps/255.)
else:
    netG = GeneratorResnet(eps=args.eps/255.)
netG.to(device)

# Optimizer
optimG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
#optimG = optim.SGD(netG.parameters(), lr=args.lr)

# Data
data_transform = transforms.Compose([
    transforms.Resize(scale_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

    return t

train_set = datasets.ImageFolder(args.train_dir, data_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
train_size = len(train_set)
print('Training data size:', train_size)

# Loss
def CWLoss(logits, target, kappa=-0., tar=False):
    target = torch.ones(logits.size(0)).type(torch.cuda.FloatTensor).mul(target.float())
    target_one_hot = Variable(torch.eye(1000).type(torch.cuda.FloatTensor)[target.long()].cuda())
    
    real = torch.sum(target_one_hot*logits, 1)
    other = torch.max((1-target_one_hot)*logits - (target_one_hot*10000), 1)[0]
    kappa = torch.zeros_like(other).fill_(kappa)
    
    if tar:
        return torch.sum(torch.max(other-real, kappa))
    else :
        return torch.sum(torch.max(real-other, kappa))
        
criterion = CWLoss

now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())   
os.mkdir(os.path.join(now))   

copyfile('train.py',os.path.join(now,'train.py'))
copyfile('generators.py',os.path.join(now,'generators.py'))

# Training
print('Label: {} \t Model: {} \t Dataset: {} \t Saving instances: {}'.format(args.target, args.model_type, args.train_dir, args.epochs))
lam_spa = 0.0001
lam_qua = 0.0001
for epoch in range(args.epochs):
    running_loss = 0
    for i, (img, gt) in enumerate(train_loader):
        img = img.to(device)
        gt = gt.to(device)

        if args.target == -1:
            label = model(normalize(img.clone().detach())).argmax(dim=-1).detach()
        else:
            label = torch.LongTensor(img.size(0))
            label.fill_(args.target)
            label = label.to(device)

        netG.train()
        optimG.zero_grad()

        adv, adv_inf, adv_0, adv_00 = netG(img)
        adv_img = adv.clone()
        
        adv_out = model(normalize(adv))
        if args.target == -1:
            # Gradient accent (Untargetted Attack)
            loss_adv = criterion(adv_out, label) 
        else:
            # Gradient decent (Targetted Attack)
            loss_adv = criterion(adv_out, label, tar=True) 
        loss_spa = torch.norm(adv_0, 1)
        bi_adv_00 = torch.where(adv_00<0.5, torch.zeros_like(adv_00), torch.ones_like(adv_00))
        loss_qua = torch.sum((bi_adv_00 - adv_00)**2)
        loss = loss_adv + lam_spa * loss_spa + lam_qua * loss_qua
            
        loss.backward()
        optimG.step()
        
        if i % 100 == 0:
            adv_0_img = torch.where(adv_0<0.5, torch.zeros_like(adv_0), torch.ones_like(adv_0)).clone().detach()
            vutils.save_image(vutils.make_grid(adv_img, normalize=True, scale_each=True), now+'/adv.png')
            vutils.save_image(vutils.make_grid(img, normalize=True, scale_each=True), now+'/org.png')
            vutils.save_image(vutils.make_grid(adv_img-img, normalize=True, scale_each=True), now+'/noise.png')
            vutils.save_image(vutils.make_grid(adv_0_img*adv_inf, normalize=True, scale_each=True), now+'/perturb.png')
            vutils.save_image(vutils.make_grid(adv_0_img, normalize=True, scale_each=True), now+'/mask.png')
            print('l0:', torch.norm(adv_0_img, 0)/args.batch_size)
      
    torch.save(netG.state_dict(), os.path.join(now,'netG_{}_{}_{}_{}.pth'.format(args.target, args.model_type, args.train_dir, epoch)))