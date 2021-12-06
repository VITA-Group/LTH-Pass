import os 
import time 
import torch 
import numpy as np 
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from lime.wrappers.scikit_image import SegmentationAlgorithm

import Metrics.lime_image as lime_image

__all__ = ['test_interpretation', 'test_interpretation_imagenet']

def test_interpretation(model, args):

    if args.dataset == 'cifar10':
        test_set = CIFAR10(args.data, train=False, transform=None, download=True)
    elif args.dataset == 'cifar100':
        test_set = CIFAR100(args.data, train=False, transform=None, download=True)

    segmentation_fn = SegmentationAlgorithm('felzenszwalb', scale=10, sigma=0.4, min_size=20)
    explainer = lime_image.LimeImageExplainer()

    # define predict function for explanation
    preprocess_transform = transforms.ToTensor()    
    def batch_predict(images):
        model.eval()
        batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
        batch = batch.cuda()
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    stablility = []
    fidelity = []
    causal = []
    # as the data sequence is random in CIFAR, we test the first ${args.image_number} images
    for idx in range(args.image_number):
        img = test_set.data[idx] # data_type = numpy array
        explanation = explainer.explain_instance(img, 
                                                batch_predict, 
                                                segmentation_fn = segmentation_fn,
                                                top_labels=1, 
                                                hide_color=0, 
                                                num_samples=1000,
                                                num_perturb=5)
        label = explanation.top_labels[0]
        stablility.append(explanation.stability_metric[label])
        fidelity.append(explanation.fidelity_metric[label])
        causal.append(explanation.causal_metric[label])    
        if idx % int(args.image_number/5) == 0:
            print('Stage[{}/{}]'.format(idx+1, args.image_number))

    return stablility, fidelity, causal

def test_interpretation_imagenet(model, args):

    test_set = ImageFolder(os.path.join(args.data, 'val'))
    explainer = lime_image.LimeImageExplainer()

    # define predict function for explanation
    print('please make sure the normalization layer contained in the model')
    preprocess_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])    
    def batch_predict(images):
        model.eval()
        batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
        batch = batch.cuda()
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    def pil_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    stablility = []
    fidelity = []
    causal = []

    select_index = [i*50 for i in range(1000)]

    for idx in select_index:
        start_time = time.time()
        path, target = test_set.samples[idx]
        print('Test image from class: {}'.format(target))
        img = pil_loader(path)
        explanation = explainer.explain_instance(np.array(img), 
                                                batch_predict,
                                                top_labels=1, 
                                                hide_color=0, 
                                                num_samples=1000,
                                                num_perturb=5)
        label = explanation.top_labels[0]
        stablility.append(explanation.stability_metric[label])
        fidelity.append(explanation.fidelity_metric[label])
        causal.append(explanation.causal_metric[label])
        end_time = time.time()
        print('Testing times = {}'.format(end_time-start_time))    

    return stablility, fidelity, causal

