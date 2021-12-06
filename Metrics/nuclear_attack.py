import os 
import time 
import copy 
import torch 
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['nuclear_norm_attack', 'test_nuc_norm', 'nuclear_norm_perturb']


def nuclear_norm_attack(model, image, target, epsilon, alpha, coef_lambda):

    # generate bernoulli noise of magnitude alpha
    delta = torch.zeros_like(image).cuda()
    delta.bernoulli_()
    delta.add_(-0.5).mul_(2*alpha)
    delta.requires_grad = True

    # attack 
    output_delta = model(image + delta)
    output_clean = model(image)

    diff_output = (output_delta - output_clean).unsqueeze(2)
    loss_norm = torch.norm(diff_output, dim=(1,2), p='nuc')
    pre_loss_norm = loss_norm.float().detach()

    loss = F.cross_entropy(output_delta, target) + coef_lambda * loss_norm.mean()
    grad0 = torch.autograd.grad(loss, delta, only_inputs=True)[0]

    delta.data.add_(epsilon * torch.sign(grad0.data))
    delta.data.clamp_(-epsilon, epsilon)
    image_adv = torch.clamp(image + delta, min=0, max=1)

    # calculate accuracy and nuclear norm of adverarial images
    output_adv = model(image_adv)
    nuc_norm = torch.norm((output_adv - output_clean).unsqueeze(2), dim=(1,2), p='nuc').float().detach()
    correctness = (torch.argmax(output_adv, dim=1) == target).float()

    return pre_loss_norm, nuc_norm, correctness

def test_nuc_norm(val_loader, model, args):

    model.eval()

    pre_sequence = []
    nuc_norm_list = []
    nuc_norm_list_pre = []

    start = time.time()
    for i, (image, target) in enumerate(val_loader):

        image = image.cuda()
        target = target.cuda()
        pre_norm, nuc_norm, correct_seq = nuclear_norm_attack(model, image, target, args.nuc_eps, args.nuc_alpha, args.nuc_lambda)
        nuc_norm_list_pre.append(pre_norm)
        nuc_norm_list.append(nuc_norm)
        pre_sequence.append(correct_seq)

        if i % args.print_freq == 0:
            end = time.time()
            print('Test: [{0}/{1}]\t'
                'Time {2:.2f}'.format(
                    i, len(val_loader), end-start))
            start = time.time()

    accuracy = torch.mean(torch.cat(pre_sequence, dim=0)).item()
    nuc = torch.mean(torch.cat(nuc_norm_list, dim=0)).item()
    nuc_pre = torch.mean(torch.cat(nuc_norm_list_pre, dim=0)).item()

    print('* Nuc Accuracy {:.4f}'.format(accuracy))
    print('* Nuc Norm {:.4f}'.format(nuc))
    print('* Pre-Nuc Norm {:.4f}'.format(nuc_pre))

    return nuc_pre, nuc, accuracy

def extract_output(val_loader, model, args):

    model.eval()

    output_list = []

    start = time.time()
    for i, (image, target) in enumerate(val_loader):

        image = image.cuda()
        target = target.cuda()
        output = model(image)
        output_list.append(output.float().detach())

        if i % args.print_freq == 0:
            end = time.time()
            print('Test: [{0}/{1}]\t'
                'Time {2:.2f}'.format(
                    i, len(val_loader), end-start))
            start = time.time()

    output_list = torch.cat(output_list, dim=0)

    return output_list

def nuclear_norm_perturb(val_loader, model, args):

    model.eval()
    original_weight = copy.deepcopy(model.state_dict())
    perturb_weight = {}

    for key in original_weight.keys():
        if 'mask' in key:
            perturb_weight[key] = copy.deepcopy(original_weight[key])
        else:
            if len(original_weight[key].size()) == 4:
                perturb_weight[key] = torch.normal(mean = original_weight[key], std = args.sigma * (original_weight[key].abs()))
            else:
                perturb_weight[key] = copy.deepcopy(original_weight[key])

    output_origin = extract_output(val_loader, model, args)
    model.load_state_dict(perturb_weight)
    output_perturb = extract_output(val_loader, model, args)

    nuc_norm = torch.norm((output_perturb - output_origin).unsqueeze(2), dim=(1,2), p='nuc').mean()
    print('* Nuclear Norm weight {:.4f}'.format(nuc_norm))

    return nuc_norm



