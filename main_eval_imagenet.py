'''
Evaluation for Tickets:
    for cifar-10 and cifar-100

'''

import os
import pdb
import time 
import pickle
import random
import shutil
import argparse
import numpy as np  

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils.setup import *
from utils.pruner import *
from Metrics.accuracy import *
from Metrics.ood_detection import *
from Metrics.calibration import *
from Metrics.Explanation import *
from Metrics.magnitude_flatness import *
from Metrics.nuclear_attack import *
from Metrics.dead_neuron import * 

from datasets import imagenet_dataloader, imagenet_ood_dataloader, imagenet_o_index, pure_imagenet_dataloader
from advertorch.utils import NormalizeByChannelMeanStd
from models.ResNet_img import resnet50_small

parser = argparse.ArgumentParser(description='PyTorch Evaluation for ImageNet')
##################################### general setting #################################################
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--arch', type=str, default=None, help='architecture')
parser.add_argument('--pretrained', type=str, default=None, help="pretrained model")
parser.add_argument('--eval_mode', type=str, default=None, help="evaluation mode")
parser.add_argument('--output_file', type=str, default='test.pt', help="the name of output file")
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--workers', type=int, default=4, help='number of workers in dataloader')

##################################### training setting #################################################
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')

##################################### Attack setting #################################################
parser.add_argument('--norm', default='linf', type=str, help='linf or l2')
parser.add_argument('--test_eps', default=(8/255), type=float, help='epsilon of attack during testing')
parser.add_argument('--test_step', default=1, type=int, help='itertion number of attack during testing')
parser.add_argument('--test_gamma', default=(8/255), type=float, help='step size of attack during testing')
parser.add_argument('--test_randinit_off', action='store_false', help='randinit usage flag (default: on)')


def main():

    global args, best_sa
    args = parser.parse_args()
    print(args)
    
    torch.cuda.set_device(int(args.gpu))
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = imagenet_dataloader(args, use_normalize=False)
    if 'small' in args.arch:
        model = resnet50_small().cuda()
    else:
        model = models.__dict__[args.arch](pretrained=False).cuda()

    if args.pretrained:
        # load pretrained model
        print('===> loading weight from {} <==='.format(args.pretrained))
        pretrained_weight = torch.load(args.pretrained, map_location = torch.device('cuda:'+str(args.gpu)))
        if 'state_dict' in pretrained_weight:
            pretrained_weight = pretrained_weight['state_dict']
        sparse_mask = extract_mask(pretrained_weight)
        if len(sparse_mask) > 0:
            prune_model_custom(model, sparse_mask)
        model.load_state_dict(pretrained_weight)
        check_sparsity(model)

    # feed normalization layer into model (for the convenient of FGSM)
    normalize_layer = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = nn.Sequential(normalize_layer, model)
    model.cuda()
    print(model[0])

    evaluation_item_list = args.eval_mode.split(',')
    print(evaluation_item_list)

    if os.path.isfile(args.output_file):
        result_dict = torch.load(args.output_file)
    else:
        result_dict = {}

    if 'accuracy' in evaluation_item_list:
        print('Evaluate standard accuracy')
        TA = test(test_loader, model, criterion, args)
        result_dict['standard-accuracy'] = TA
        torch.save(result_dict, args.output_file)

    # test robustness
    if 'robustness' in evaluation_item_list:
        # mask sure the normalizer contains in the network layers
        show_robust_parameters(args)
        RA = test_adv(test_loader, model, criterion, args)
        result_dict['robust-accuracy'] = RA
        torch.save(result_dict, args.output_file)

    if 'corruption' in evaluation_item_list: 
        # Evaluate on corrupted dataset
        print('Evaluate standard accuracy on corrupted dataset')
        corrupt_acc = {}
        imagenet_c_path = args.data + '-c'
        print('Using corrupted dataset in ', imagenet_c_path)
        
        file_list = os.listdir(imagenet_c_path)
        file_list.sort()

        for file_name in file_list:
            corrupt_acc[file_name] = []
            for severity in range(1,6):
                print('attack_type={}'.format(file_name), 'severity={}'.format(severity))
                
                imagenet_c_test_loader = torch.utils.data.DataLoader(
                    datasets.ImageFolder(os.path.join(imagenet_c_path, file_name, str(severity)), transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor()
                    ])),
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

                CTA = test(imagenet_c_test_loader, model, criterion, args)
                corrupt_acc[file_name].append(CTA)
        result_dict['corruption-accuracy'] = corrupt_acc
        torch.save(result_dict, args.output_file)

    if 'ood' in evaluation_item_list: 

        in_loader, out_loader = imagenet_ood_dataloader(args, use_normalize=False)
        label_mask = imagenet_o_index()

        aurocs, auprs = test_ood_imagenet(model, in_loader, out_loader, args, label_mask)

        result_dict['OoD-detection-auroc'] = aurocs
        result_dict['OoD-detection-aupr'] = auprs
        torch.save(result_dict, args.output_file)


    if 'calibration' in evaluation_item_list:
        print('Evaluation for calibration')  
        calibration_ece = {}
        calibration_nll = {}
        ece, nll = test_calibration(test_loader, model, args)
        calibration_ece['origin'] = ece 
        calibration_nll['origin'] = nll 

        imagenet_c_path = args.data + '-c'
        print('Using corrupted dataset in ', imagenet_c_path)
        
        file_list = os.listdir(imagenet_c_path)
        file_list.sort()

        for file_name in file_list:
            calibration_ece[file_name] = []
            calibration_nll[file_name] = []
            for severity in range(1,6):
                print('attack_type={}'.format(file_name), 'severity={}'.format(severity))

                imagenet_c_test_loader = torch.utils.data.DataLoader(
                    datasets.ImageFolder(os.path.join(imagenet_c_path, file_name, str(severity)), transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor()
                    ])),
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

                ece, nll = test_calibration(imagenet_c_test_loader, model, args)
                calibration_ece[file_name].append(ece)
                calibration_nll[file_name].append(nll)

        result_dict['calibration'] = calibration_ece
        result_dict['nll'] = calibration_nll
        torch.save(result_dict, args.output_file)

    if 'pac_bayes_weight' in evaluation_item_list:

        train_loader = pure_imagenet_dataloader(args, use_normalize=False)

        max_sigma_weight = pac_bayes_mag_flat_ce(model, train_loader, beta=0.1, iteration_times=10, max_search_times=15, sigma_max=2)
        print('* max sigma respect to weight = {}'.format(max_sigma_weight))
        result_dict['pac_bayes'] = max_sigma_weight
        torch.save(result_dict, args.output_file)

    if 'pac_bayes_input' in evaluation_item_list:

        train_loader = pure_imagenet_dataloader(args, use_normalize=False)

        max_sigma_input = pac_bayes_input_flat_ce(model, train_loader, beta=0.1, iteration_times=3 , max_search_times=15, sigma_max=2)
        print('* max sigma respect to input = {}'.format(max_sigma_input))
        result_dict['pac_bayes_input'] = max_sigma_input
        torch.save(result_dict, args.output_file)

    if 'interpretation' in evaluation_item_list:
        print('Evaluation Interpretation based on LIME')
        stablility, fidelity, causal = test_interpretation_imagenet(model, args)
        result_dict['Interpertation-stablility'] = stablility
        result_dict['Interpertation-fidelity'] = fidelity
        result_dict['Interpertation-causal'] = causal
        torch.save(result_dict, args.output_file)


if __name__ == '__main__':
    main()




