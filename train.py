# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.cuda.amp import autocast,GradScaler
from datasets.make_dataloader import make_dataset
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from PIL import Image
import time
import os
import yaml
from shutil import copyfile
from utils import get_model_list, load_network, save_network, make_weights_for_balanced_classes
from optimizers.make_optimizer import make_optimizer
from losses.triplet_loss import Tripletloss,TripletLoss
from losses.cal_loss import cal_kl_loss,cal_loss,cal_triplet_loss
from models.model import make_model
import sys
import numpy as np
sys.path.append('./models/')
import ReCU
version =  torch.__version__
#fp16
try:
    from apex.fp16_utils import *
    from apex import amp
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--dataset',default='sue', type=str,help='datasets choose from thise list ["sue","university"]')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='convnext_tri', type=str, help='output model name')
parser.add_argument('--optimizer', type=str,
        choices=['Adam', 'SGD', 'Lion','AdamLion'],
        help="Choose from the list:['Adam', 'SGD', 'Lion','AdamLion']")
parser.add_argument('--bigradual_n1', default = 20, help="n1 parameter of bifradual unfreezing") 
parser.add_argument('--bigradual_n2', default = 20, help="n2 parameter of bifradual unfreezing")      
parser.add_argument('--data_dir',default='../data/train',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_false', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
parser.add_argument('--pad', default=10, type=int, help='padding')
parser.add_argument('--h', default=384, type=int, help='height')
parser.add_argument('--w', default=384, type=int, help='width')
parser.add_argument('--views', default=2, type=int, help='the number of views')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.0, type=float, help='drop rate')
parser.add_argument('--DA', action='store_false', help='use Color Data Augmentation' )
parser.add_argument('--resnet', action='store_true', default=True, help='use resnet' )
parser.add_argument('--share', action='store_false',default=True, help='share weight between different view' )
parser.add_argument('--resume', action='store_true', help='use resume trainning' )
parser.add_argument('--autocast', action='store_true',default=True, help='use mix precision' )
parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory' )
parser.add_argument('--block', default=2, type=int, help='')
parser.add_argument('--kl_loss', action='store_true',default=False, help='kl_loss' )
parser.add_argument('--triplet_loss', default=0.3, type=float, help='')
parser.add_argument('--sample_num', default=1, type=float, help='')
parser.add_argument('--model', default='recunet34', type=str, metavar='MODEL', help='Name of model to train')
parser.add_argument('--epochs', default=100, type=int, help='' )
parser.add_argument('--fname', default='train.txt', type=str, help='Name of log txt')
parser.add_argument('--steps', default=[80,120,200], type=int, help='' )
parser.add_argument('--print_flips', action='store_true',default=False, help='use mix precision' )
opt = parser.parse_args()

dir_name = os.path.join('./model',opt.name)
if not opt.resume:
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
#record every run
    copyfile('./train.py', dir_name+'/train.py')
    copyfile('models/ConvNext/backbones/model_convnext.py', dir_name + '/model.py')


if opt.resume:
    model, opt, start_epoch = load_network(opt.name, opt)
else:
    start_epoch = 0

str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
######################################################################
# Load Data
# ---------
dataloaders,class_names,dataset_sizes = make_dataset(opt)
opt.nclasses = len(class_names)
print(dataset_sizes)
if not opt.resume:
    with open(os.path.join('model',opt.name,opt.fname),'a',encoding='utf-8') as f:
        text = str(dataset_sizes)+'\n'
        f.write(text)
use_gpu = torch.cuda.is_available()

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

    
y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []
def set_requires_grad_true(model):
    """
    Set requires_grad to True for all parameters in a PyTorch model.

    Args:
        model (nn.Module): The model whose parameters will be modified.
    """
    for param in model.parameters():
        param.requires_grad = True
def count_weight_flips(model, prev_weights):
    flips = 0
    for param, prev_param in zip(model.parameters(), prev_weights.parameters()):
        flips += torch.sum((param.data.sign() != prev_param.sign()).float()).item()
    return flips
def train_model(model,teacher, opt, model_test,optimizer_ft,exp_lr_scheduler, bin_optimizer,bin_exp_lr_scheduler, num_epochs=25):
    since = time.time()

    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()
    loss_kl = nn.KLDivLoss(reduction='batchmean')
    triplet_loss = Tripletloss(margin=opt.triplet_loss)
    
    min_loss = 18.0

    warm_up = 0.1 # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['satellite']/opt.batchsize)*opt.warm_epoch # first 5 epoch
    def cpt_tau(epoch):
        "compute tau"
        a = torch.tensor(np.e)
        T_min, T_max = torch.tensor(0.85).float(), torch.tensor(0.99).float()
        A = (T_max - T_min) / (a - 1)
        B = T_min - A
        tau = A * torch.tensor([torch.pow(a, epoch/200)]).float() + B
        return tau 

    #* record names of conv_modules
    conv_modules=[]
    for name, module in model.named_modules():
        if isinstance(module,ReCU.imagenet.modules.binarized_modules.BinarizeConv2d):
            conv_modules.append(module)
    flips_count = []
    
    for epoch in range(num_epochs-start_epoch):
        flips_current = 0
        if epoch == int(opt.bigradual_n1): 
            set_requires_grad_true(model)
        epoch = epoch + start_epoch
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        tau = cpt_tau(epoch)
        for module in conv_modules:
            module.tau = tau.cuda()
        with open(os.path.join('model',opt.name,opt.fname),'a',encoding='utf-8') as f:
            text = str('Epoch {}/{}'.format(epoch, num_epochs - 1))+'\n'+('-' * 10)+'\n'
            f.write(text)
        teacher.train(True)
        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_cls_loss = 0.0
            running_triplet = 0.0
            running_kl_loss = 0.0
            running_loss = 0.0
            running_corrects = 0.0
            running_corrects2 = 0.0
            running_corrects3 = 0.0
            # Iterate over data.
            for data,data2,data3 in dataloaders:
                # satallite # street # drone
                loss = 0.0
                # get the inputs
                inputs, labels = data
                inputs2, labels2 = data2
                inputs3, labels3 = data3
                now_batch_size,c,h,w = inputs.shape
                if now_batch_size<opt.batchsize: # skip the last batch
                    continue
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    inputs2 = Variable(inputs2.cuda().detach())
                    inputs3 = Variable(inputs3.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                    labels2 = Variable(labels2.cuda().detach())
                    labels3 = Variable(labels3.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
 
                # zero the parameter gradients
                
                optimizer_ft.zero_grad()
                bin_optimizer.zero_grad()
                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs, outputs2 = model(inputs, inputs3)
                else:
                    with autocast():
                        if opt.views == 2:
                            outputs, outputs2 = model(inputs, inputs3)  # satellite and drone
                            with torch.no_grad():
                                outputs_t, outputs2_t = teacher(inputs, inputs3)
                                
                            # print(len(outputs))
                        elif opt.views == 3:
                            outputs, outputs3, outputs2 = model(inputs, inputs2, inputs3)# satellite1 and drone2 and street3
                f_triplet_loss = torch.tensor((0))
                number = 0
                loss_KD_teacher = nn.MSELoss()( torch.stack(outputs[0]), torch.stack(outputs_t[0])) \
                                 + nn.MSELoss()( torch.stack(outputs2[0]), torch.stack(outputs2_t[0]))\
                                 + nn.MSELoss()( torch.stack(outputs[1]), torch.stack(outputs_t[1])) \
                                + nn.MSELoss()( torch.stack(outputs2[1]), torch.stack(outputs2_t[1]))
                                
                 
                #for _ in range(len( outputs[0])):
                #   loss_KD_teacher += nn.MSELoss()( outputs[0][_],outputs_t[0][_]) +nn.MSELoss()( outputs2[0][_],outputs2_t[0][_]) 
                 
                #loss_KD_teacher += nn.MSELoss()( outputs[0][0],outputs_t[0][0]) +nn.MSELoss()( outputs2[0][0],outputs2_t[0][0]) 
                
                loss_KD_teacher /= 4
                if opt.triplet_loss>0:
                    features = outputs[1]
                    features2 = outputs2[1]
                    split_num = opt.batchsize//opt.sample_num
                    f_triplet_loss = cal_triplet_loss(features,features2,labels,triplet_loss,split_num)

                    outputs = outputs[0]
                    outputs2 = outputs2[0]

                if isinstance(outputs,list):
                    preds = []
                    preds2 = []
                    for out,out2 in zip(outputs,outputs2):
                        preds.append(torch.max(out.data,1)[1])
                        preds2.append(torch.max(out2.data,1)[1])
                else:
                    _, preds = torch.max(outputs.data, 1)
                    _, preds2 = torch.max(outputs2.data, 1)

                kl_loss = torch.tensor((0))
                if opt.views == 2:
                    cls_loss = cal_loss(outputs, labels, criterion) + cal_loss(outputs2, labels3, criterion)
                    if opt.kl_loss:
                        kl_loss = cal_kl_loss(outputs, outputs2, loss_kl)
                elif opt.views == 3:
                    outputs3 = outputs3[0]
                    if isinstance(outputs, list):
                        preds3 = []
                        for out3 in outputs3:
                            preds3.append(torch.max(out3.data, 1)[1])
                        cls_loss = cal_loss(outputs, labels, criterion) + cal_loss(outputs2, labels3, criterion) + cal_loss(outputs3,labels2,criterion)
                        loss += cls_loss
                    else:
                        _, preds3 = torch.max(outputs3.data, 1)
                        cls_loss = cal_loss(outputs, labels, criterion) + cal_loss(outputs2, labels3, criterion) + cal_loss(outputs3,labels2,criterion)
                        loss += cls_loss

                loss = f_triplet_loss #+ loss_KD_teacher*2
                # backward + optimize only if in training phase
                if epoch<opt.warm_epoch and phase == 'train': 
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss *= warm_up

                if phase == 'train':
                    import pickle
                    if opt.print_flips:
                        old_model = pickle.loads(pickle.dumps(model))
                    if opt.autocast:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer_ft)
                        if epoch >int(opt.bigradual_n2): #n2 = 30
                            scaler.step(bin_optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer_ft.step()
                        if epoch > int(opt.bigradual_n2):
                            bin_optimizer.step()
                    if opt.print_flips:
                        flips_current +=count_weight_flips(model,old_model)
                    
                    ##########

                # statistics
                if int(version[0]) > 0 or int(version[2]) > 3:  # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * now_batch_size
                    running_cls_loss += cls_loss.item() * now_batch_size
                    running_triplet += f_triplet_loss.item() * now_batch_size
                    running_kl_loss += kl_loss.item() * now_batch_size
                else:  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                    running_cls_loss += cls_loss.data[0] * now_batch_size
                    running_triplet += f_triplet_loss.data[0] * now_batch_size
                    running_kl_loss += kl_loss.data[0] * now_batch_size


                if isinstance(preds,list) and isinstance(preds2,list):
                    running_corrects += sum([float(torch.sum(pred == labels.data)) for pred in preds])/len(preds)
                    if opt.views==2:
                        running_corrects2 += sum([float(torch.sum(pred == labels3.data)) for pred in preds2]) / len(preds2)
                    else:
                        running_corrects2 += sum([float(torch.sum(pred == labels3.data)) for pred in preds2])/len(preds2)
                else:
                    running_corrects += float(torch.sum(preds == labels.data))
                    if opt.views == 2:
                        running_corrects2 += float(torch.sum(preds2 == labels3.data))
                    else:
                        running_corrects2 += float(torch.sum(preds2 == labels3.data))
                if opt.views == 3:
                    if isinstance(preds,list) and isinstance(preds2,list):
                        running_corrects3 += sum([float(torch.sum(pred == labels2.data)) for pred in preds3])/len(preds3)
                    else:
                        running_corrects3 += float(torch.sum(preds3 == labels2.data))

            if opt.print_flips:
                print("flips:",flips_current)
                flips_count.append(flips_current)
            epoch_cls_loss = running_cls_loss / dataset_sizes['satellite']
            epoch_kl_loss = running_kl_loss / dataset_sizes['satellite']
            epoch_triplet_loss = running_triplet / dataset_sizes['satellite']
            epoch_loss = running_loss / dataset_sizes['satellite']
            epoch_acc = running_corrects / dataset_sizes['satellite']
            epoch_acc2 = running_corrects2 / dataset_sizes['satellite']


            lr_backbone = optimizer_ft.state_dict()['param_groups'][0]['lr']
            lr_other = optimizer_ft.state_dict()['param_groups'][0]['lr']
            if opt.views == 2:
                print('{} Loss: {:.4f} Cls_Loss:{:.4f} KL_Loss:{:.4f} Triplet_Loss {:.4f} Satellite_Acc: {:.4f}  Drone_Acc: {:.4f} lr_backbone:{:.6f} lr_other {:.6f}'
                                                                                .format(phase, epoch_loss,epoch_cls_loss,epoch_kl_loss,epoch_triplet_loss, epoch_acc, epoch_acc2,lr_backbone,lr_other))
                with open(os.path.join('model', opt.name, opt.fname), 'a', encoding='utf-8') as f:
                    text = str('{} Loss: {:.4f} Cls_Loss:{:.4f} KL_Loss:{:.4f} Triplet_Loss {:.4f} Satellite_Acc: {:.4f}  Drone_Acc: {:.4f}'
                                                                                .format(phase, epoch_loss,epoch_cls_loss,epoch_kl_loss,epoch_triplet_loss, epoch_acc, epoch_acc2)) + '\n'
                    f.write(text)
            elif opt.views == 3:
                epoch_acc3 = running_corrects3 / dataset_sizes['satellite']
                print('{} Loss: {:.4f} Cls_Loss:{:.4f} KL_Loss:{:.4f} Triplet_Loss {:.4f} Satellite_Acc: {:.4f}  Street_Acc: {:.4f} Drone_Acc: {:.4f} lr_backbone:{:.6f} lr_other {:.6f}'
                                                                                .format(phase,epoch_loss,epoch_cls_loss,epoch_kl_loss,epoch_triplet_loss, epoch_acc, epoch_acc3, epoch_acc2, lr_backbone,lr_other))
                with open(os.path.join('model', opt.name, opt.fname), 'a', encoding='utf-8') as f:
                    text = str('{} Loss: {:.4f} Cls_Loss:{:.4f} KL_Loss:{:.4f} Triplet_Loss {:.4f} Satellite_Acc: {:.4f}  Street_Acc: {:.4f} Drone_Acc: {:.4f} lr_backbone:{:.6f} lr_other {:.6f}'
                                                                                .format(phase,epoch_loss,epoch_cls_loss,epoch_kl_loss,epoch_triplet_loss, epoch_acc, epoch_acc3, epoch_acc2, lr_backbone,lr_other)) + '\n'
                    f.write(text)


            
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)            
            # deep copy the model
            if phase == 'train':
                exp_lr_scheduler.step()
                if epoch >30:

                    bin_exp_lr_scheduler.step()
            # if epoch > 90 and epoch%10 == 9:
            #     save_network(model, opt.name, epoch)
            #     min_loss = min(min_loss, epoch_loss)
            if epoch >= 90 and epoch_loss < min_loss:
                save_network(model, opt.name, epoch)
                min_loss = epoch_loss
            #draw_curve(epoch)
        if opt.print_flips:                   
            print(flips_count)
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()
        with open(os.path.join('model',opt.name,opt.fname), 'a', encoding='utf-8') as f:
            text = str('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) + '\n'
            f.write(text)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    with open(os.path.join('model',opt.name,opt.fname), 'a', encoding='utf-8') as f:
        text = str('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) + '\n'
        f.write(text)
    #print('Best val Acc: {:4f}'.format(best_acc))
    #save_network(model_test, opt.name+'adapt', epoch)

    return model


######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join('./model',opt.name,'train.jpg'))


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#
import copy
opt_teacher = copy.deepcopy(opt)
opt_teacher.name = "convnext_tri"
opt_teacher.model = "convnext_small_22k_224"
opt_teacher.resnet=False
if not opt.resume:
    teacher = make_model(opt_teacher)
    teacher.load_state_dict(opt.torch.load(opt.teacher_path))
    model = make_model(opt)
    #if 
    #current_state_dict = model.state_dict()
    #checkpoint = torch.load("./university_tiny.pth")
    #filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in current_state_dict and current_state_dict[k].size() == v.size()}
    #for k, v in filtered_checkpoint.items():
    #    v.requires_grad = False
    #model.load_state_dict(filtered_checkpoint,strict = False)
    
    # save opt
    with open('%s/opt.yaml'%dir_name,'a') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)
# print(model)


# For resume:
if start_epoch>=40:
    opt.lr = opt.lr*0.01

# optimizer_ft = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
# # Decay LR by a factor of 0.1 every 40 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
optimizer_ft,exp_lr_scheduler, bin_optimizer,bin_exp_lr_scheduler = make_optimizer(model,opt)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU.
# model to gpu
model = model.cuda()
teacher = teacher.cuda()
if opt.fp16:
    model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level = "O1")

model_test = None
num_epochs = opt.epochs
def conv_flops_counter_hook(conv_module, input, output, extra_per_position_flops=0):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims, dtype=np.int64)) * \
        (in_channels * filters_per_channel + extra_per_position_flops)

    active_elements_count = batch_size * int(np.prod(output_dims, dtype=np.int64))

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:

        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops
    print(overall_flops)
    conv_module.__flops__ += int(overall_flops) /64

from ptflops import get_model_complexity_info
from models.BNext.src.bnext import HardBinaryConv

#from models.ReCU.imagenet.modules import *__class__
#classss = model.model_1.convnext.model.layer1[0].conv1.#classss:conv_flops_counter_hook
#print(classss)
import torch
from torchvision import models, transforms
from PIL import Image

# Load a pre-trained ResNet-50 model
#model = models.resnet50(pretrained=True)
#model.eval()
def count_parameters(model):
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

    # Populate param_to_module dictionary
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            param_to_module[param] = module

    # Count parameters based on the mapping
    for param in model.parameters():
        total_params += param.numel()
        if isinstance(param_to_module.get(param, None), HardBinaryConv):
            hbc_params += param.numel()

    # Calculate parameters not in HardBinaryConv modules
    non_hbc_params = total_params - hbc_params

#    return total_params, hbc_params, non_hbc_params
#total_params, hbc_params, non_hbc_params  = count_parameters(model)
#print(f"Total Parameters: {total_params}")
#print(f"Parameters in HardBinaryConv: {hbc_params}")
#print(f"Parameters not in HardBinaryConv: {non_hbc_params}")
#assert False
#with torch.cuda.device(0):
#  macs, params = get_model_complexity_info(model, (3, 384, 384), as_strings=True,
 #                                          print_per_layer_stat=True, verbose=True,custom_modules_hooks={ HardBinaryConv:conv_flops_counter_hook})
#  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#  print('{:<30}  {:<8}'.format('Number of parameters: ', params))
#assert False
model = train_model(model,teacher, opt, model_test, optimizer_ft,exp_lr_scheduler, bin_optimizer,bin_exp_lr_scheduler,
                       num_epochs=num_epochs)


