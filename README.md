# A Screening Test for Sparse Neural Networks: What We Lose When We Prune



## Environment:

​	pytorch

​	torchvision

​	advertorch



## Usage:

1. Eval CIFAR-10/100 

```
python -u main_eval.py \
	--data [data-direction] \
	--dataset cifar10 \ # choose from [cifar10, cifar100]
	--arch resnet20s \ # choose from [resnet20s,resnet18,wideresnet]
	--pretrained [pretrained-weight] \ 
	--eval_mode accuracy,robustness,corruption,ood,calibration,interpretation,pac_bayes_weight,pac_bayes_input \
	--output_file result.pt \
	--test_randinit_off \
	--image_number 1000 
```

2. Eval Imagenet 

```
python -u main_eval_imagenet.py \
    --data [data-direction] \
    --arch resnet50 \
    --pretrained [pretrained-weight] \
    --eval_mode accuracy,robustness,corruption,ood,calibration,interpretation,pac_bayes_weight,pac_bayes_input \
    --output_file result.pt 
```

3. Eval Hessian

```
python -u vis_pyhessian_analysis.py \
    --data [data-direction] \
    --dataset [dataset] \ choose from [cifar10, cifar100, imagenet]
    --arch [network-architecture] \ choose from [resnet20s,resnet18,wideresnet,resnet50]
    --pretrained [pretrained-weight] \
    --output_file result.pt \
    --mode weight,input
```

4. Composition neurons

cd composition-neuron, modified from https://github.com/jayelm/compexp

