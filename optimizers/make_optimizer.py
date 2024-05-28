import torch.optim as optim
from torch.optim import lr_scheduler
def split_parameters(model):
    """
    Count the total number of parameters in a PyTorch model, the number of parameters in HardBinaryConv modules,
    and the number of parameters not in HardBinaryConv modules.

    Parameters:
    model (torch.nn.Module): The PyTorch model to analyze.

    Returns:
    tuple: A tuple containing the total number of parameters, the number in HardBinaryConv modules, and
           the number not in HardBinaryConv modules.
    """
    total_params = 0
    hbc_params = 0

    # Mapping parameter ownership
    param_to_module = {}
    bin=[]
    non_bin=[]
    # Populate param_to_module dictionary
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            param_to_module[param] = module

    # Count parameters based on the mapping
    for param in model.parameters():
        total_params += param.numel()
        if isinstance(param_to_module.get(param, None), HardBinaryConv):
            bin.append(param)
        else:
            non_bin.append(param)
    return bin,non_bin
from models.BNext.src.bnext import HardBinaryConv
def make_optimizer(model,opt):
    ignored_params = []
    if opt.views==3:
        for i in [model.model_1, model.model_2]:
            ignored_params += list(map(id, i.convnext.parameters()))
    else:
        for i in [model.model_1]:
            ignored_params += list(map(id, i.convnext.parameters()))
    extra_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    base_params = filter(lambda p: id(p) in ignored_params, model.parameters())
    bin,nonbin = split_parameters(model)
    import sys
    sys.path.append('.')
    from lion_pytorch import Lion
    if opt.optimizer.lower() == "adam":
        bin_optimizer = optim.Adam([{'params': bin, 'lr': opt.lr/6},])
        fp_optimizer = optim.Adam( [{'params': nonbin, 'lr': opt.lr}])
    elif opt.optimizer.lower() == "sgd":
        bin_optimizer = optim.SGD([{'params': bin, 'lr': opt.lr/6},])
        fp_optimizer = optim.SGD( [{'params': nonbin, 'lr': opt.lr}])
    elif opt.optimizer.lower() == "lion":
        bin_optimizer = Lion([{'params': bin, 'lr': opt.lr/6},])
        fp_optimizer = Lion( [{'params': nonbin, 'lr': opt.lr}])
    elif opt.optimizer.lower() == "adamlion":
        bin_optimizer = Lion([{'params': bin, 'lr': opt.lr/6},])
        fp_optimizer = optim.Adam( [{'params': nonbin, 'lr': opt.lr}])
    else:
        print("Optimizer not found",opt.optimizer.lower() )
        assert False
    bin_exp_lr_scheduler = lr_scheduler.MultiStepLR(bin_optimizer, milestones=opt.steps, gamma=0.1)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(fp_optimizer, milestones=opt.steps, gamma=0.1)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=80, gamma=0.1)
    # exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.95)
    # exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, patience=4, verbose=True,threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08)

    return fp_optimizer,exp_lr_scheduler, bin_optimizer,bin_exp_lr_scheduler