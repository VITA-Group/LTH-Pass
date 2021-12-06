
from advertorch.utils import NormalizeByChannelMeanStd
from datasets import cifar10_dataloaders, cifar100_dataloaders
from models.ResNet_cifar import resnet20, resnet20_small
from models.ResNet import resnet18, resnet18_small, resnet50
from models.WideResNet_cifar import WideResNet, WideResNet_small

__all__ = ['setup_model_dataset', 'show_robust_parameters']

def setup_model_dataset(args):
    
    if args.dataset == 'cifar10':
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_loader, val_loader, test_loader = cifar10_dataloaders(batch_size=args.batch_size, data_dir=args.data)

    elif args.dataset == 'cifar100':
        classes = 100
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673,	0.2564,	0.2762])
        train_loader, val_loader, test_loader = cifar100_dataloaders(batch_size=args.batch_size, data_dir=args.data)
    
    else:
        raise ValueError('unknow dataset')

    if args.arch == 'resnet20s':
        print('build model resnet20_cifar')
        model = resnet20(number_class=classes)
    elif args.arch == 'resnet18':
        print('build model resnet18')
        model = resnet18(num_classes=classes, imagenet=False)
    elif args.arch == 'resnet50':
        print('build model resnet50')
        model = resnet50(num_classes=classes, imagenet=False)
    elif args.arch == 'wideresnet':
        print('build model: wideresnet-28-10')
        model = WideResNet(depth=28, num_classes=classes, widen_factor=10, dropRate=0.0)

    elif args.arch == 'resnet20s_small':
        print('build model resnet20_cifar')
        model = resnet20_small(number_class=classes)
    elif args.arch == 'resnet18_small':
        print('build model resnet18')
        model = resnet18_small(num_classes=classes, imagenet=False)
    elif args.arch == 'wideresnet_small':
        print('build model: wideresnet-28-10')
        model = WideResNet_small(depth=28, num_classes=classes, widen_factor=10, dropRate=0.0)

    else:
        raise ValueError('unknow model')

    model.normalize = normalization

    return model, train_loader, val_loader, test_loader


def show_robust_parameters(args):
    print('* evaluation robustness')
    print('* Adversarial settings')
    print('* norm = {}'.format(args.norm))
    print('* eps = {}'.format(args.test_eps))
    print('* steps = {}'.format(args.test_step))
    print('* alpha = {}'.format(args.test_gamma))
    print('* randinit = {}'.format(args.test_randinit_off))
