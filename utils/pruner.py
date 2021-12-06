import copy 
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def pruning_model(model, px):

    print('start unstructured pruning')
    parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m,'weight'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def pruning_model_random(model, px):

    parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m,'weight'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=px,
    )

def prune_model_custom(model, mask_dict):

    print('start unstructured pruning with custom mask')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])

def remove_prune(model):
    
    print('remove pruning')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.remove(m,'weight')

def extract_mask(model_dict):

    new_dict = {}

    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = copy.deepcopy(model_dict[key])

    return new_dict

def check_sparsity(model):
    
    sum_list = 0
    zero_sum = 0

    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            sum_list = sum_list+float(m.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(m.weight == 0))  

    print('* remain weight = ', 100*(1-zero_sum/sum_list),'%')
    
    return 100*(1-zero_sum/sum_list)

def check_sparsity_mask(mask_dict):

    sum_list = 0
    zero_sum = 0

    for key in mask_dict.keys():
        sum_list = sum_list+float(mask_dict[key].nelement())
        zero_sum = zero_sum+float(torch.sum(mask_dict[key] == 0))  

    print('* remain weight = ', 100*(1-zero_sum/sum_list),'%')
    
    return 100*(1-zero_sum/sum_list)

def pruning_with_rewind(model, px, init, mode='l1'):

    if mode == 'l1':
        print('using L1 magnitude pruning')
        pruning_model(model, px)
    elif mode == 'random':
        print('using random pruning')
        pruning_model_random(model, px)
    else:
        raise ValueError('Unsupport pruning mode')

    current_mask = extract_mask(model.state_dict())
    remove_prune(model)

    model.load_state_dict(init)
    prune_model_custom(model, current_mask)
    check_sparsity(model)

