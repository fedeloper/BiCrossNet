import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
import timm
m = timm.create_model('seresnet50', pretrained=True)
m.eval()
with torch.cuda.device(0):
  net = m
  macs, params = get_model_complexity_info(net, (3, 256, 256), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))