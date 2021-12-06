
import os
import time 
import torch
import random
import shutil
import numpy as np  
import torch.nn as nn 
import sklearn.metrics as sk
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from datasets import imagenet_o_index


__all__ = ['test_ood', 'test_ood_imagenet']

def extract_prediction(val_loader, model, args, mask=None):

    model.eval()
    start = time.time()

    y_pred = []
    y_true = []

    for i, (input, target) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda()
    
        # compute output
        with torch.no_grad():
            output = model(input)[:,mask]
            number_image = output.shape[0]
            output = output.reshape(number_image, -1)
            pred = F.softmax(output, dim=1)

        y_pred.append(pred.cpu().numpy())
        y_true.append(target.cpu().numpy())

        if i % args.print_freq == 0:
            end = time.time()
            print('Predict-OoD: [{0}/{1}]\t'
                'Time {2:.2f}'.format(
                    i, len(val_loader), end-start))
            start = time.time()
    
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    print('prediction shape = ', y_pred.shape)
    print('ground truth shape = ', y_true.shape)

    return y_pred, y_true


def test_ood(model, in_loader, out_loader, args):

    # in distribution
    y_pred_ind, _ = extract_prediction(in_loader, model, args)
    ind_labels = np.ones(y_pred_ind.shape[0])
    ind_scores = np.max(y_pred_ind, 1)

    # out of distribution
    y_pred_ood, _ = extract_prediction(out_loader, model, args)
    ood_labels = np.zeros(y_pred_ood.shape[0])
    ood_scores = np.max(y_pred_ood, 1)

    labels = np.concatenate([ind_labels, ood_labels])
    scores = np.concatenate([ind_scores, ood_scores])

    auroc = sk.roc_auc_score(labels, scores)
    print('* AUROC = {}'.format(auroc))
    return auroc


def test_ood_imagenet(model, in_loader, out_loader, args, label_mask=None):

    # in distribution
    y_pred_ind, _ = extract_prediction(in_loader, model, args, mask=label_mask)
    ind_labels = np.zeros(y_pred_ind.shape[0])
    ind_scores = -np.max(y_pred_ind, 1)

    # out of distribution
    y_pred_ood, _ = extract_prediction(out_loader, model, args, mask=label_mask)
    ood_labels = np.ones(y_pred_ood.shape[0])
    ood_scores = -np.max(y_pred_ood, 1)

    labels = np.concatenate([ind_labels, ood_labels])
    scores = np.concatenate([ind_scores, ood_scores])

    auroc = sk.roc_auc_score(labels, scores)
    aupr = sk.average_precision_score(labels, scores)
    print('* AUROC = {}'.format(auroc))
    print('* AUPR = {}'.format(aupr))
    return auroc, aupr









