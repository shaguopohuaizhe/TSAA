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
import torch.nn.functional as F

from generators import GeneratorResnet
import time

parser = argparse.ArgumentParser(description='Test sparse')
parser.add_argument('--test_dir', default="input path to validation data here", help='ImageNet Validation Data')
parser.add_argument('--model_type', type=str, default='incv3',  help='Model against GAN is trained: incv3, res50')
parser.add_argument('--model_t',type=str, default='res50',  help='Model under attack : vgg16, incv3, res50, dense161')
parser.add_argument('--target', type=int, default=-1, help='-1 if untargeted')
parser.add_argument('--eps', type=int, default=255, help='Perturbation Budget')
parser.add_argument('--batch_size', type=int, default=10, help='Batch Size')
parser.add_argument('--checkpoint', type=str, default='',  help='path to checkpoint')
args = parser.parse_args()
print(args)

# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Input dimensions: Inception takes 3x299x299
if args.model_type in ['res50']:
    scale_size = 256
    img_size = 224
else:
    scale_size = 300
    img_size = 299

if args.model_type == 'incv3':
    netG = GeneratorResnet(inception=True, eps=args.eps/255., evaluate=True)
else:
    netG = GeneratorResnet(eps=args.eps/255., evaluate=True)
netG.load_state_dict(torch.load(args.checkpoint))
netG.to(device)
netG.eval()

if args.model_t == 'dense161':
    model_t = torchvision.models.densenet161(pretrained=True)
elif args.model_t == 'vgg16':
    model_t = torchvision.models.vgg16(pretrained=True)
elif args.model_t == 'incv3':
    model_t = torchvision.models.inception_v3(pretrained=True)
elif args.model_t == 'res50':
    model_t = torchvision.models.resnet50(pretrained=True)
model_t = model_t.to(device)
model_t.eval()

def trans_incep(x):
    x = F.interpolate(x, size=(299,299), mode='bilinear', align_corners=False)
    return x

# Setup-Data
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

test_dir = args.test_dir
test_set = datasets.ImageFolder(test_dir, data_transform)
test_size = len(test_set)
print('Test data size:', test_size)

test_loader = torch.utils.data.DataLoader(test_set,batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Evaluation
adv_acc = 0
clean_acc = 0
fool_rate = 0
target_rate = 0
norm = 0
time_count = 0

for i, (img, label) in enumerate(test_loader):
    img, label = img.to(device), label.to(device)

    if 'inc' in args.model_t or 'xcep' in args.model_t:
        clean_out = model_t(normalize(trans_incep(img.clone().detach())))
    elif 'denoise' in args.model_t:
        clean_out = model_t(img.clone().detach())
    else:
        clean_out = model_t(normalize(img.clone().detach()))
    clean_acc += torch.sum(clean_out.argmax(dim=-1) == label).item()
    #print(clean_out.argmax(dim=-1))

    # Adversary
    start = time.time()
    adv,_,adv_0,adv_00 = netG(img)
    end = time.time()
    times = end - start
    time_count += times

    if 'inc' in args.model_t or 'xcep' in args.model_t:
        adv_out = model_t(normalize(trans_incep(adv.clone().detach())))
    elif 'denoise' in args.model_t:
        adv_out = model_t(adv.clone().detach())
    else:
        adv_out = model_t(normalize(adv.clone().detach()))
    adv_acc +=torch.sum(adv_out.argmax(dim=-1) == label).item()

    fool_rate += torch.sum(adv_out.argmax(dim=-1) != clean_out.argmax(dim=-1)).item()
    #print(adv_out.argmax(dim=-1))
    
    if args.target != -1:
        target = torch.LongTensor(img.size(0))
        target.fill_(args.target)
        target = target.to(device)
        target_rate += torch.sum(adv_out.argmax(dim=-1) == target).item()

    norm += torch.norm(adv_0.clone().detach(), 0)
    
print('L0 norm:', norm/test_size)
print('time:', time_count/test_size)
if args.target != -1:
    print('Clean:{0:.3%}\t Adversarial :{1:.3%}\t Fooling Rate:{2:.3%}\t Target Success Rate:{3:.3%}'.format(clean_acc/test_size, adv_acc/test_size, fool_rate/test_size, target_rate/test_size))
else:
    print('Clean:{0:.3%}\t Adversarial :{1:.3%}\t Fooling Rate:{2:.3%}'.format(clean_acc/test_size, adv_acc/test_size, fool_rate/test_size))

