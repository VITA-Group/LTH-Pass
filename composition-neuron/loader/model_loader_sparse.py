import settings
import torch
import torchvision
import utils_pruning

def noop(*args, **kwargs):
    pass


def loadmodel(
    hook_fn,
    feature_names=settings.FEATURE_NAMES,
    hook_modules=None,
    pretrained_override=None,
):
    device = torch.device("cuda" if settings.GPU else "cpu")
    model_fn = torchvision.models.__dict__[settings.MODEL]

    if settings.MODEL_FILE is None:
        model = model_fn(
            pretrained=pretrained_override if pretrained_override is not None else True
        )
    elif settings.MODEL_FILE == "<UNTRAINED>":
        model = model_fn(
            pretrained=pretrained_override if pretrained_override is not None else False
        )
    else:
        checkpoint = torch.load(settings.MODEL_FILE, map_location='cpu')
        print('load model from {}'.format(settings.MODEL_FILE))
        if (
            type(checkpoint).__name__ == "OrderedDict"
            or type(checkpoint).__name__ == "dict"
        ):
            model = model_fn(num_classes=settings.NUM_CLASSES)
            if settings.MODEL_PARALLEL:
                state_dict = {
                    str.replace(k, "module.", ""): v
                    for k, v in checkpoint["state_dict"].items()
                }  # the data parallel layer will add 'module' before each layer name
            else:
                if 'state_dict' in checkpoint.keys():
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint

            #extract mask
            current_mask = utils_pruning.extract_mask(state_dict)
            print('mask length', len(current_mask))
            if len(current_mask):
                print('pruning')
                utils_pruning.prune_model_custom(model, current_mask)
            for key in state_dict.keys():
                assert key in model.state_dict().keys()
            model.load_state_dict(state_dict)
            utils_pruning.check_sparsity(model)
        else:
            if settings.MODEL == 'densenet161':
                # Fix old densenet pytorch names.
                model = model_fn(num_classes=settings.NUM_CLASSES)
                state_dict = checkpoint.state_dict()

                def rep(k):
                    for i in range(6):
                        k = k.replace(f"norm.{i}", f"norm{i}")
                        k = k.replace(f"relu.{i}", f"relu{i}")
                        k = k.replace(f"conv.{i}", f"conv{i}")
                    return k

                state_dict = {
                    rep(k): v for k, v in state_dict.items()
                }
                model.load_state_dict(state_dict)
            else:
                model = checkpoint
    if hook_fn is not None:
        for name in feature_names:
            if isinstance(name, list):
                # Iteratively retrive the module
                hook_model = model
                for n in name:
                    hook_model = hook_model._modules.get(n)
            else:
                hook_model = model._modules.get(name)
            if hook_model is None:
                raise ValueError(f"Couldn't find feature {name}")
            if hook_modules is not None:
                hook_modules.append(hook_model)
            hook_model.register_forward_hook(hook_fn)
    if settings.GPU:
        model.cuda()
    model.eval()
    return model



