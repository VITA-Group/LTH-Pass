import os 
import torch 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F

__all__ = ['test_dead_and_freeze_neuron']

'''
https://github.com/evcu/pytorchpruner/blob/master/pytorchpruner/unitscorers.py
'''

def norm_i(tensr):
    n_units = tensr.size(1)
    norm_arr = torch.zeros(n_units)
    for ui in range(n_units):
        norm_arr[ui] = tensr[:,ui,].norm(p=1)
    return norm_arr.cpu()

class meanOutputReplacer(torch.nn.Module):
    def __init__(self,module,unit_id=0):
        """
        meanOutputReplacer warps any module single output module.
        This module has two purpose(mode) and they can be switched with the flags 'enabled' and 'is_mean_replace'
        @params enabled Enables the module, if not enabled the forward is equal to the result of the wrapped Module
        @params is_mean_replace if the flag `enabled` is also true, the output `unit_id`th unit replaced with its mean.
            if not true: than only the zero_mean output is calculated, to be able to calculate the MRS score efficiently
            on the back-pass-hook.
        """
        super(meanOutputReplacer, self).__init__()
        if not isinstance(unit_id,int):
            raise ValueError('not implemented must be int')
            ## note that implementing for whole layer is straight forward, just define bunch of 1's
        if isinstance(module, meanOutputReplacer):
            raise ValueError("ERROR: module given is a meanOutputReplacer, this should be wrong")

        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.enabled = False
        self.is_mean_replace = False
        self.unit_id = unit_id
        def backwardhook(l,inp,out):
            if l.enabled and not l.is_mean_replace:
                prdct = (out[0].data*l.cy_zeromean)
                self.mrss = norm_i(prdct)
                self.freezes = norm_i(out[0].data)
        self.register_backward_hook(backwardhook)

    def forward(self,*inputs, **kwargs):
        if len(inputs)>1:
            raise ValueError('meanOutputReplacer is not implemented for layer getting multiple inputs')
        if self.enabled:
            out = self.module(*inputs, **kwargs)

            out_mean = out.mean(0)
            # second dim is the n_outputs
            while out_mean.dim()>1:
                out_mean = out_mean.mean(1)
            self.cy_mean = out_mean.data
            # import pdb;pdb.set_trace()
            if self.is_mean_replace:
                if isinstance(self.unit_id,torch.ByteTensor):
                    ## CAN DO THIS MORE EFFICIENTLY USING TENSOR OPS
                    for i in range(len(self.unit_id)):
                        if self.unit_id[i]==1:
                            out[:,i] = out_mean[i].expand(out.size(0),*out.size()[2:])
                else:
                    out[:,self.unit_id] = out_mean[self.unit_id].expand(out.size(0),*out.size()[2:])
            else:
                out_mean_expanded = out_mean.data.expand(out.size(0),
                                                        *out.size()[2:],
                                                        -1)
                if out_mean_expanded.dim()>2:
                    out_mean_expanded= out_mean_expanded.transpose(1,3)
                self.cy_zeromean = out.data-out_mean_expanded
        else:
            out = self.module(*inputs, **kwargs)
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(\n\t' \
            + 'module=' + str(self.module) \
            + '\n\t,is_mean_replace=' + str(self.is_mean_replace) \
            + '\n\t,enabled=' + str(self.enabled) + ')'

def modify_conv(model):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = modify_conv(model=module)

        if isinstance(module, nn.Conv2d):
            model._modules[name] = meanOutputReplacer(module)

    return model

def test_dead_and_freeze_neuron(model, datalodaer, args):

    # modified model with meanOutputReplacer
    model.eval()
    model = modify_conv(model)

    # enable for conv layers
    for name, module in model.named_modules():
        if isinstance(module, meanOutputReplacer):
            module.enabled = True
            module.is_mean_replace = False

    # mini-batch images
    image, target = next(iter(datalodaer))
    image = image.cuda()
    target = target.cuda()

    # one forward and backward step 
    output = model(image)
    loss = F.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()

    statics = {}
    mrs_zero_all = 0
    freeze_zero_all = 0 
    element_all = 0

    # show MRS and Freeze scores
    for name, module in model.named_modules():
        if isinstance(module, meanOutputReplacer):
            statics[name] = {}
            kernel_num = module.weight[0].nelement()
            mrs_zero = kernel_num * (module.mrss == 0).float().sum().item()
            freeze_zero = kernel_num * (module.freezes == 0).float().sum().item()
            element = module.weight.nelement()
            print(name, 'MRS-sparsity = {:.2f}, Freeze-sparsity = {:.2f}'.format(100*mrs_zero/element, 100*freeze_zero/element))
            mrs_zero_all += mrs_zero
            freeze_zero_all += freeze_zero
            element_all += element
            statics[name]['MRS-all'] = [mrs_zero, element, kernel_num, 100*mrs_zero/element]
            statics[name]['MRS'] = 100*mrs_zero/element
            statics[name]['Freeze-all'] = [freeze_zero, element, kernel_num, 100*freeze_zero/element]
            statics[name]['Freeze'] = 100*freeze_zero/element

    print('Overall: MRS-sparsity = {:.2f}, Freeze-sparsity = {:.2f}'.format(100*mrs_zero_all/element_all, 100*freeze_zero_all/element_all))
    statics['Overall'] = {}
    statics['Overall']['MRS'] = 100*mrs_zero_all/element_all
    statics['Overall']['Freeze'] = 100*freeze_zero_all/element_all

    return statics






