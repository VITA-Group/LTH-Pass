#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

from __future__ import print_function

import os
import sys
import json
import time 
import random 
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

from pyhessian import hessian, hessian_input
from process.pyhessian_density_plot import get_esd_plot

from utils.pruner import *
import torchvision.models as models
from models.ResNet_cifar import resnet20, resnet20_small
from models.ResNet import resnet18, resnet18_small
from models.WideResNet_cifar import WideResNet, WideResNet_small
from advertorch.utils import NormalizeByChannelMeanStd
from datasets import pure_cifar10_dataloaders, pure_cifar100_dataloaders, pure_imagenet_dataloader
from models.ResNet_img import resnet50_small


# Settings
parser = argparse.ArgumentParser(description='PyTorch pyhessian analysis')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--batch_size', type=int, default=200, help='batch size')
parser.add_argument('--batch_number', type=int, default=1, help='Data number = batch_size*batch_number')

parser.add_argument('--arch', type=str, default='resnet20', help='architecture')
parser.add_argument('--pretrained', type=str, default=None, help='pretrained weight')
parser.add_argument('--output_file', type=str, default='test.pt', help="the name of output file")
parser.add_argument('--mode', type=str, default='weight')

parser.add_argument('--workers', type=int, default=4, help='number of workers in dataloader')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
args = parser.parse_args()


torch.cuda.set_device(args.gpu)



# set random seed to reproduce the work
torch.manual_seed(args.seed) 
torch.cuda.manual_seed_all(args.seed) 
np.random.seed(args.seed) 
random.seed(args.seed) 
torch.backends.cudnn.deterministic = True 

for arg in vars(args):
    print(arg, getattr(args, arg))

if os.path.isfile(args.output_file):
    result_dict = torch.load(args.output_file)
else:
    result_dict = {}

#############################################################################
#########################  Get the hessian data  ############################
#############################################################################

if args.dataset == 'cifar10':
    print('Dataset = CIFAR10')
    classes = 10
    normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    train_loader = pure_cifar10_dataloaders(args.batch_size, args.data)
elif args.dataset == 'cifar100':
    print('Dataset = CIFAR100')
    classes = 100
    normalization = NormalizeByChannelMeanStd(
        mean=[0.5071, 0.4866, 0.4409], std=[0.2673,	0.2564,	0.2762])
    train_loader = pure_cifar100_dataloaders(args.batch_size, args.data)
elif args.dataset == 'imagenet':
    print('Dataset = ImageNet')
    train_loader = pure_imagenet_dataloader(args, use_normalize=True)
else:
    raise ValueError(('unsupport dataset'))

if args.batch_number == 1:
    for inputs, labels in train_loader:
        hessian_dataloader = (inputs, labels)
        break
else:
    hessian_dataloader = []
    for i, (inputs, labels) in enumerate(train_loader):
        hessian_dataloader.append((inputs, labels))
        if i == args.batch_number - 1:
            break

#############################################################################
##############################  Get the model  ##############################
#############################################################################

if 'cifar' in args.dataset:
    if args.arch == 'resnet20s':
        print('build model resnet20_cifar')
        model = resnet20(number_class=classes)
    elif args.arch == 'resnet18':
        print('build model resnet18')
        model = resnet18(num_classes=classes, imagenet=False)
    elif args.arch == 'wideresnet':
        print('build model: wideresnet-28-10')
        model = WideResNet(depth=28, num_classes=classes, widen_factor=10, dropRate=0.0)
    elif args.arch == 'resnet20s_small':
        print('build model resnet20_cifar_small')
        model = resnet20_small(number_class=classes)
    elif args.arch == 'resnet18_small':
        print('build model resnet18_small')
        model = resnet18_small(num_classes=classes, imagenet=False)
    elif args.arch == 'wideresnet_small':
        print('build model: wideresnet-28-10_small')
        model = WideResNet_small(depth=28, num_classes=classes, widen_factor=10, dropRate=0.0)
    else:
        raise ValueError('unknow model')

    model.normalize = normalization

else:
    if 'small' in args.arch:
        print('build resnet50 small')
        model = resnet50_small()
    else:
        print('build resnet50')
        model = models.resnet50(pretrained=False)

model.cuda()

if args.pretrained:
    print('===> loading weight from {} <==='.format(args.pretrained))
    pretrained_weight = torch.load(args.pretrained, map_location = torch.device('cuda:'+str(args.gpu)))
    if 'state_dict' in pretrained_weight:
        pretrained_weight = pretrained_weight['state_dict']
    sparse_mask = extract_mask(pretrained_weight)
    if len(sparse_mask) > 0:
        prune_model_custom(model, sparse_mask)
    model.load_state_dict(pretrained_weight)
    check_sparsity(model)

#############################################################################
##########################  Begin the computation  ##########################
#############################################################################

if 'weight' in args.mode:

    print('hessian analysis respect to weight')
    criterion = nn.CrossEntropyLoss()

    model.eval()
    if args.batch_number == 1:
        hessian_comp = hessian(model, criterion, data=hessian_dataloader)
    else:
        hessian_comp = hessian(model, criterion, dataloader=hessian_dataloader)

    print('********** finish data loading and begin Hessian computation **********')
    start = time.time()
    top_eigenvalues, _ = hessian_comp.eigenvalues()
    end = time.time()

    trace = hessian_comp.trace()

    end2 = time.time()

    print(end - start)
    print(end2 - end)

    print('\n***Top Eigenvalues: ', top_eigenvalues)
    print('\n***Trace: ', np.mean(trace))
    result_dict['eigenvalues'] = top_eigenvalues
    result_dict['trace'] = trace
    torch.save(result_dict, args.output_file)


##########################################################################################
##############  Begin the computation hessian respect to input  ##########################
##########################################################################################
if 'input' in args.mode:

    print('hessian analysis respect to input')
    criterion = nn.CrossEntropyLoss()

    model.eval()
    if args.batch_number == 1:
        hessian_comp_input = hessian_input(model, criterion, data=hessian_dataloader)
    else:
        hessian_comp_input = hessian_input(model, criterion, data=hessian_dataloader[0]) 

    print('********** finish data loading and begin Hessian computation **********')
    start = time.time()
    top_eigenvalues_input, _ = hessian_comp_input.eigenvalues()
    end = time.time()

    trace_input = hessian_comp_input.trace()

    end2 = time.time()
    print('\n***Top Eigenvalues (input): ', top_eigenvalues_input)
    print('\n***Trace (input): ', np.mean(trace_input))
    result_dict['eigenvalues_input'] = top_eigenvalues_input
    result_dict['trace_input'] = trace_input
    print(end - start)
    print(end2 - end)

    torch.save(result_dict, args.output_file)
