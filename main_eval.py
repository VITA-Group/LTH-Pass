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

from datasets import cifar_c_dataloaders, ood_cifar10, ood_cifar100, ood_svhn, ood_gaussion, pure_cifar10_dataloaders, pure_cifar100_dataloaders

parser = argparse.ArgumentParser(description='PyTorch Evaluation for CIFAR')
##################################### general setting #################################################
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset for evaluation')
parser.add_argument('--arch', type=str, default=None, help='architecture')
parser.add_argument('--pretrained', type=str, default=None, help="pretrained model")
parser.add_argument('--eval_mode', type=str, default=None, help="evaluation mode")
parser.add_argument('--output_file', type=str, default='test.pt', help="the name of output file")
parser.add_argument('--image_number', default=100, type=int, help='image numbers for Interpretation')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')

##################################### training setting #################################################
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')

##################################### Attack setting #################################################
parser.add_argument('--norm', default='linf', type=str, help='linf or l2')
parser.add_argument('--test_eps', default=(8/255), type=float, help='epsilon of attack during testing')
parser.add_argument('--test_step', default=1, type=int, help='itertion number of attack during testing')
parser.add_argument('--test_gamma', default=(8/255), type=float, help='step size of attack during testing')
parser.add_argument('--test_randinit_off', action='store_false', help='randinit usage flag (default: on)')

##################################### Nuclear Norm setting #################################################
parser.add_argument('--nuc_eps', default=(8/255), type=float, help='epsilon for nuclear norm attack')
parser.add_argument('--nuc_alpha', default=(4/255), type=float, help='alpha for nuclear norm attack')
parser.add_argument('--nuc_lambda', default=0.01, type=float, help='lambda for nuclear norm attack')
parser.add_argument('--sigma', default=0.05, type=float, help='weight perturbation level')

def main():

    global args, best_sa
    args = parser.parse_args()
    print(args)
    
    torch.cuda.set_device(int(args.gpu))
    criterion = nn.CrossEntropyLoss()

    #setup model&dataset for evaluation
    model, train_loader, val_loader, test_loader = setup_model_dataset(args)
    model.cuda()
    #load pretrained model
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
        if args.dataset == 'cifar10':
            cifar_c_path = os.path.join(args.data, 'CIFAR-10-C')
        elif args.dataset == 'cifar100':
            cifar_c_path = os.path.join(args.data, 'CIFAR-100-C')

        file_list = os.listdir(cifar_c_path)
        file_list.sort()

        for file_name in file_list:
            if not file_name == 'labels.npy':
                attack_type = file_name[:-len('.npy')]
                corrupt_acc[attack_type] = []
                for severity in range(1,6):
                    print('attack_type={}'.format(attack_type), 'severity={}'.format(severity))
                    cifar_c_loader = cifar_c_dataloaders(args.batch_size, cifar_c_path, severity, attack_type)
                    acc = test(cifar_c_loader, model, criterion, args)
                    corrupt_acc[attack_type].append(acc)
        result_dict['corruption-accuracy'] = corrupt_acc
        torch.save(result_dict, args.output_file)

    if 'ood' in evaluation_item_list: 
        ood_data_name = ['svhn', 'gaussian']
        ood_result = {}
        # evaluate Out of distribution detection performance
        out_loader1 = ood_svhn(args.batch_size, args.data)
        out_loader2 = ood_gaussion(args.batch_size, 10000)

        if args.dataset == 'cifar10':
            out_loader3 = ood_cifar100(args.batch_size, args.data)
            ood_data_name.append('cifar100')
        elif args.dataset == 'cifar100':
            out_loader3 = ood_cifar10(args.batch_size, args.data)
            ood_data_name.append('cifar10')
        out_loader_list = [out_loader1, out_loader2, out_loader3]

        for i, out_loader in enumerate(out_loader_list):
            auroc = test_ood(model, test_loader, out_loader, args)
            ood_result[ood_data_name[i]] = auroc
        
        result_dict['OoD-detection'] = ood_result
        torch.save(result_dict, args.output_file)

    if 'calibration' in evaluation_item_list:
        print('Evaluation for calibration')  
        calibration_ece = {}
        calibration_nll = {}
        ece, nll = test_calibration(test_loader, model, args)
        calibration_ece['origin'] = ece 
        calibration_nll['origin'] = nll 

        #calibration on corrupted datasets
        if args.dataset == 'cifar10':
            cifar_c_path = os.path.join(args.data, 'CIFAR-10-C')
        elif args.dataset == 'cifar100':
            cifar_c_path = os.path.join(args.data, 'CIFAR-100-C')

        file_list = os.listdir(cifar_c_path)
        file_list.sort()

        for file_name in file_list:
            if not file_name == 'labels.npy':
                attack_type = file_name[:-len('.npy')]
                calibration_ece[attack_type] = []
                calibration_nll[attack_type] = []
                for severity in range(1,6):
                    print('attack_type={}'.format(attack_type), 'severity={}'.format(severity))
                    cifar_c_loader = cifar_c_dataloaders(args.batch_size, cifar_c_path, severity, attack_type)
                    ece, nll = test_calibration(cifar_c_loader, model, args)
                    calibration_ece[attack_type].append(ece)
                    calibration_nll[attack_type].append(nll)
        result_dict['calibration'] = calibration_ece
        result_dict['nll'] = calibration_nll
        torch.save(result_dict, args.output_file)

    if 'interpretation' in evaluation_item_list:
        print('Evaluation Interpretation based on LIME')
        stablility, fidelity, causal = test_interpretation(model, args)
        result_dict['Interpertation-stablility'] = stablility
        result_dict['Interpertation-fidelity'] = fidelity
        result_dict['Interpertation-causal'] = causal
        torch.save(result_dict, args.output_file)

    if 'pac_bayes_weight' in evaluation_item_list:
        
        if args.dataset == 'cifar10':
            train_loader = pure_cifar10_dataloaders(args.batch_size, args.data)
        elif args.dataset == 'cifar100':
            train_loader = pure_cifar100_dataloaders(args.batch_size, args.data)
        else:
            raise ValueError('unsupport dataset')

        max_sigma_weight = pac_bayes_mag_flat_ce(model, train_loader, beta=0.1)
        print('* max sigma weight = {}'.format(max_sigma_weight))

        result_dict['pac_bayes'] = max_sigma_weight
        torch.save(result_dict, args.output_file)

    if 'pac_bayes_input' in evaluation_item_list:
        
        if args.dataset == 'cifar10':
            train_loader = pure_cifar10_dataloaders(args.batch_size, args.data)
        elif args.dataset == 'cifar100':
            train_loader = pure_cifar100_dataloaders(args.batch_size, args.data)
        else:
            raise ValueError('unsupport dataset')

        max_sigma_input = pac_bayes_input_flat_ce(model, train_loader, beta=0.1, iteration_times=3 , max_search_times=15, sigma_max=2)
        print('* max sigma input = {}'.format(max_sigma_input))

        result_dict['pac_bayes_input'] = max_sigma_input
        torch.save(result_dict, args.output_file)


    if 'nuclear_norm' in evaluation_item_list:
        print('Evaluate Nuclear Norm')
        pre_nuc, nuc, accuracy = test_nuc_norm(test_loader, model, args)
        result_dict['nuclear-norm'] = nuc
        result_dict['nuclear-norm_pre'] = pre_nuc
        result_dict['nuclear-accuracy'] = accuracy
        weight_nuc = nuclear_norm_perturb(test_loader, model, args)
        result_dict['nuclear-norm-weight'] = weight_nuc
        torch.save(result_dict, args.output_file) 
    
    if 'dead_neuron' in evaluation_item_list:
        result = test_dead_and_freeze_neuron(model, test_loader, args)
        result_dict['dead-neuron'] = result['Overall']['MRS']
        result_dict['freeze-neuron'] = result['Overall']['Freeze']
        torch.save(result_dict, args.output_file) 

if __name__ == '__main__':
    main()




